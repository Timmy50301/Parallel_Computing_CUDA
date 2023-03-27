# Monte Carlo Simulation with CUDA Parallel Computing

It is an implemtation of parallel computing with CUDA on a simple simulation with Monte Carlo Method. Comparison between multi-thread GPU and single/multi core CPU is made.  

## Basic Usage

```bash
$ make # see makefile for details
```
```bash
$ ./main
```

## Configurations

```bash
const int TOTAL_POINTS=1000000;     // 做1000000次樣本蒐集
```
```bash
const int CORE=4;                   // 使用四個CPU核心，若機器沒有四核心則會自動排程
```
```bash
const int NUM=3;                    // 做三個不同thread數量的實驗
```
```bash
const int GRID[3]={10,50,100}; 
const int THREAD[3]={10,50,100};    // thread數量{10*10, 50*50, 100*100}
```
```bash
const int SEED1=66;
const int SEED2=666;
const int SEED3=7777;               // 生成亂數的變數
```

