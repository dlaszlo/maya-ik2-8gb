#!/usr/bin/env python3
"""Replay recorded routing through the engine's actual ExpertCache, with CUDA copies stubbed.
No GPU timing is simulated. Report CPU-computed experts, admissions, and policy CPU cost.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('trace', type=Path)
a = p.parse_args()
source = (ROOT / 'src/ik2/ik2_main.cu').read_text()
cache = source[source.index('struct ExpertCache {'):source.index('// ---- the forward')]
preamble = r'''
#include "ik2_cache_policy.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <cmath>
#include <string>
#include <vector>
#include <sys/mman.h>
using cudaStream_t = void*;
void *g_s=(void*)1,*g_cs=(void*)2,*g_ls=nullptr;
uint64_t g_seq_done=0,g_seq_next=1;
void lazy_poll() {}
int cudaStreamSynchronize(void*) {g_seq_done=g_seq_next-1;return 0;}
int cudaMemcpyAsync(void*,const void*,size_t,int,void*) {return 0;}
constexpr int cudaMemcpyHostToDevice=0;
#define CU(x) ((void)(x))
[[noreturn]] void die(const std::string& s) {fprintf(stderr,"%s\n",s.c_str());abort();}
template<class T> T* dalloc(size_t n) {
 void* p=mmap(nullptr,n*sizeof(T),PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
 if(p==MAP_FAILED) die("mmap"); return (T*)p;
}
struct Layer {
 int dev=0; size_t eb[3],stride[3],slot_bytes;
 const uint8_t* host[3]={}; bool host_contig=false;
};
size_t tp_bytes(const Layer& l,int) {return l.slot_bytes;}
void tp_part(const Layer&,int,int,size_t&,size_t&) {abort();}
'''
main = r'''
struct Record { int l,T,K; std::vector<int> picks; };
void check(const ExpertCache& ec) {
 for(size_t p=0;p<ec.pools.size();++p) {
  const auto& P=ec.pools[p]; std::vector<bool> seen(P.n_slots);
  for(int q=1;q<=2;++q) {
   int prev=-1, count=0;
   for(int s=q==1?P.prob_head:P.main_head;s>=0;s=P.next[s]) {
    assert(s<P.n_active && !seen[s]); seen[s]=true;
    assert(P.qid[s]==q && P.prev[s]==prev && P.owner[s]>=0);
    assert(ec.loc[P.owner[s]]==s && ec.layer_pool[P.owner[s]/ec.E]==(int)p);
    prev=s; ++count;
   }
   assert(count==(q==1?P.prob_sz:P.main_sz));
   assert(prev==(q==1?P.prob_tail:P.main_tail));
  }
  for(int s=P.free_head;s>=0;s=P.next[s]) {
   assert(s<P.n_active && !seen[s] && P.owner[s]<0 && P.qid[s]==0); seen[s]=true;
  }
  for(int s=0;s<P.n_active;++s) assert(seen[s]);
  for(int s=P.n_active;s<P.n_slots;++s) assert(!seen[s] && P.owner[s]<0);
 }
 for(int i=0;i<(int)ec.loc.size();++i)
  if(ec.loc[i]>=0) assert(ec.pools[ec.layer_pool[i/ec.E]].owner[ec.loc[i]]==i);
}
int main(int argc,char**argv) {
 std::ifstream f(argv[1],std::ios::binary); assert(f);
 std::vector<Record> records;
 for(;;) {
  int hdr[3]; f.read((char*)hdr,sizeof(hdr));
  if(f.gcount()==0) break; assert(f.gcount()==sizeof(hdr));
  Record r{hdr[0],hdr[1],hdr[2],{}};
  assert(r.l>=0 && r.l<24 && r.T>0 && r.T<=16384 && r.K==8);
  r.picks.resize(r.T*r.K); f.read((char*)r.picks.data(),r.picks.size()*4); assert(f);
  for(int e:r.picks) assert(e>=0 && e<128);
  records.push_back(std::move(r));
 }
 std::vector<int> schedule;
 for(int i=0;i<6;++i) schedule.push_back(i);
 for(int i=0;i<36;++i) schedule.push_back(6+i%12);
 for(int i=18;i<24;++i) schedule.push_back(i);
 std::vector<Layer> layers(24);
 for(int l=0;l<24;++l) {
  for(int j=0;j<3;++j) layers[l].eb[j]=layers[l].stride[j]=(size_t)2304*1152*(l<20?36:34)/(l<20?64:32);
  layers[l].slot_bytes=3*layers[l].eb[0];
 }
 for(int delay:{0,2}) for(int mode:{0,1}) {
  ExpertCache ec; g_seq_done=0;g_seq_next=1;
  ec.init(layers,0,24,128,439*layers[0].slot_bytes+144*layers[20].slot_bytes);
  assert(ec.pools[0].n_slots==439 && ec.pools[1].n_slots==144);
  ec.schedule.init(schedule,24,128,8); ec.schedule.enabled=mode;
  long visits=0,cpu=0,uploads=0,skips=0; double micros=0;
  long cpu_half[2]={},up_half[2]={};
  std::deque<std::pair<long,uint64_t>> pending;
  long total_visits=0;for(auto&r:records)if(r.T==1)++total_visits;
  for(const auto& r:records) {
   ec.schedule.active=false; const uint64_t tick=++ec.tick;
   if(r.T!=1) {
    g_seq_done=g_seq_next-1; pending.clear(); ec.scan=true;
    std::vector<bool> picked(128);
    for(int e:r.picks) picked[e]=true;
    for(int e=0;e<128;++e) if(picked[e]) ec.get(layers[r.l],r.l,e,tick,0,nullptr,false,true);
    ec.scan=false;check(ec);continue;
   }
   assert(r.l==schedule[visits%48]);
   while(!pending.empty() && pending.front().first<=visits) {
    g_seq_done=std::max(g_seq_done,pending.front().second);pending.pop_front();
   }
   const auto start=std::chrono::steady_clock::now();
   ec.observe_decode(visits%48,r.l,r.picks.data(),8,tick);
   int added=0;uint64_t seq=0;const int half=visits<total_visits/2?0:1;
   for(int e:r.picks) {
    if(ec.has(r.l,e)) {
     ec.get(layers[r.l],r.l,e,tick,0,nullptr,false,false);
     if(ec.slot_sseq(ec.last_handle)>g_seq_done) ++cpu,++cpu_half[half];
    } else {
     ++cpu; ++cpu_half[half];
     bool admit=ec.worth_admitting(r.l,e,tick);
     if(!admit) ++skips;
     if(admit && added<1 && pending.size()<2) {
      ec.get(layers[r.l],r.l,e,tick,0,nullptr,false,false);
      if(!seq) seq=g_seq_next++;
      ec.slot_sseq(ec.last_handle)=seq;
      ++uploads;++up_half[half];++added;
     } else ++ec.misses;
    }
   }
   if(seq) pending.push_back({visits+delay+1,seq});
   micros+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
   ++visits;check(ec);
  }
  double tokens=visits/48.0;
  printf("%s delay=%d tokens=%.0f CPU_experts/token=%.3f copies/token=%.3f policy_us/token=%.1f admission_skips=%ld halves_cpu=%.2f/%.2f halves_copies=%.2f/%.2f\n",
   mode?"schedule":"fifo",delay,tokens,cpu/tokens,uploads/tokens,micros/tokens,skips,
   cpu_half[0]/(tokens/2),cpu_half[1]/(tokens/2),up_half[0]/(tokens/2),up_half[1]/(tokens/2));
  // Protect both a current pick and an in-flight copy, even if their priority is lowest.
  ec.schedule.active=true; auto& P=ec.pools[0];
  int live=-1; for(int s=0;s<P.n_active;++s)if(P.owner[s]>=0){live=s;break;}
  assert(live>=0);P.used[live]=ec.tick+1;
  assert(ec.schedule_victim(P,ec.tick+1)!=live);
  P.used[live]=0;P.sseq[live]=g_seq_next+10;
  assert(ec.schedule_victim(P,ec.tick+1)!=live);
  g_seq_done=g_seq_next+10;
  ec.lend(0,100);check(ec);
  int victim=ec.schedule_victim(P,ec.tick+1);assert(victim< P.n_active);
  ec.give_back(0);check(ec);
  for(auto&pool:ec.pools)munmap(pool.data,pool.slot_bytes*pool.n_slots);
 }
 puts("PASS: queue/location invariants, current-pick and in-flight protection, lend/give_back");
}
'''
with tempfile.TemporaryDirectory(prefix='ik2-cache-replay-') as tmp:
    tmp = Path(tmp)
    (tmp / 'replay.cpp').write_text(preamble + cache + main)
    subprocess.run(['g++-13', '-O3', '-std=c++20', '-I'+str(ROOT/'src/ik2'), str(tmp/'replay.cpp'), '-o', str(tmp/'replay')], check=True)
    subprocess.run([str(tmp/'replay'), str(a.trace.resolve())], check=True)
