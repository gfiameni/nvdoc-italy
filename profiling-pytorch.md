# This short guide provides basic instructions on how to profile a PyTorch code using different tools, including NVIDIA NSIGHT SYSTEM

## Some links
* https://developer.nvidia.com/nsight-systems
* https://docs.nvidia.com/nsight-systems/profiling/index.html

## How to correctly measure time spent on the GPU

```
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
z = x + y
end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print(start.elapsed_time(end))
```


## Define NVTX Regions

* https://docs.nvidia.com/nvtx/index.html

```
# In your script, write
# torch.cuda.nvtx.range_push("region name")
# ...
# torch.cuda.nvtx.range_pop()
# around suspected hotspot regions for easy identification on the timeline.
#
# Dummy/warmup iterations prior to the region you want to profile are highly
# recommended to get caching allocator/cuda context initialization out of the way.

# Focused profiling, profiles only a target region
# (your app must call torch.cuda.cudart().cudaProfilerStart()/Stop() at the start/end of the target region)
```

## nsys
```
# Typical use (collects GPU timeline, Cuda and OS calls on the CPU timeline, but no CPU stack traces)
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o nsight_report -f true -x true python script.py args...

# Adds CPU backtraces that will show when you mouse over a long call or small orange tick (sample) on the CPU timeline:
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report -f true --cudabacktrace=true --cudabacktrace-threshold=10000 --osrt-threshold=10000 -x true python script.py args...


nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report -f true --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true --cudabacktrace-threshold=10000 --osrt-threshold=10000 -x true python script.py args...

# if appname creates child processes, nsys WILL profile those as well.  They will show up as separate processes with
# separate timelines when you open the profile in nsight-sys

# Breakdown of options:
nsys profile
-w true # Don't suppress app's console output.
-t cuda,nvtx,osrt,cudnn,cublas # Instrument, and show timeline bubbles for, cuda api calls, nvtx ranges,
                               # os runtime functions, cudnn library calls, and cublas library calls.
                               # These options do not require -s cpu nor do they silently enable -s cpu.
-s cpu # Sample the cpu stack periodically.  Stack samples show up as little tickmarks on the cpu timeline.
       # Last time i checked they were orange, but still easy to miss.
       # Mouse over them to show the backtrace at that point.
       # -s cpu can increase cpu overhead substantially (I've seen 2X or more) so be aware of that distortion.
       # -s none disables cpu sampling.  Without cpu sampling, the profiling overhead is reduced.
       # Use -s none if you want the timeline to better represent a production job (api calls and kernels will
       # still appear on the profile, but profiling them doesn't distort the timeline nearly as much).
-o nsight_report # output file
-f true # overwrite existing output file
--capture-range=cudaProfilerApi # Only start profiling when the app calls cudaProfilerStart...
--stop-on-range-end=true # ...and end profiling when the app calls cudaProfilerStop.
--cudabacktrace=true # Collect a cpu stack sample for cuda api calls whose runtime exceeds some threshold.
                     # When you mouse over a long-running api call on the timeline, a backtrace will
                     # appear, and you can identify which of your functions invoked it.
                     # I really like this feature.
                     # Requires -s cpu.
--cudabacktrace-threshold=10000 # Threshold (in nanosec) that determines how long a cuda api call
                                # must run to trigger a backtrace.  10 microsec is a reasonable value
                                # (most kernel launches should take less than 10 microsec) but you
                                # should retune if you see a particular api call you'd like to investigate.
                                # Requires --cudabacktrace=true and -s cpu.
--osrt-threshold=10000 # Threshold (in nanosec) that determines how long an os runtime call (eg sleep)
                       # must run to trigger a backtrace.
                       # Backtrace collection for os runtime calls that exceed this threshold should
                       # occur by default if -s cpu is enabled.
-x true # Quit the profiler when the app exits.
python script.py args...
```

## cProfile
* https://docs.python.org/3/library/profile.html

## PyTorch Profiler
* https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

## PyTorch Lightning Profiler
* https://pytorch-lightning.readthedocs.io/en/1.5.10/advanced/profiler.html

## Scalene
* https://github.com/emeryberger/scalene

## Python Line Profiler
* https://github.com/pyutils/line_profiler


```
# Copy paste the desired command and run it for your app. It will produce a .qdrep file.
# Run the "nsight-sys" GUI executable and File->Open the .qdrep file.
# If you're making the profile locally on your desktop, you may not need nsys at all, you can do
# the whole workflow (create and view profile) through the GUI, but if your job runs remotely on
# a cluster node, I prefer to create .qdrep profiles with nsys remotely, copy them back to my desktop,
# then open them in nsight-sys.
```

## PyTorch Multiprocessing Best Practices
* https://pytorch.org/docs/stable/notes/multiprocessing.html

## How long does it take to load the torch library?

```
python -X importtime -c "import torch"
```

```
-X opt : set implementation-specific option. The following options are available:
         -X faulthandler: enable faulthandler
         -X showrefcount: output the total reference count and number of used
             memory blocks when the program finishes or after each statement in the
             interactive interpreter. This only works on debug builds
         -X tracemalloc: start tracing Python memory allocations using the
             tracemalloc module. By default, only the most recent frame is stored in a
             traceback of a trace. Use -X tracemalloc=NFRAME to start tracing with a
             traceback limit of NFRAME frames
         -X importtime: show how long each import takes. It shows module name,
             cumulative time (including nested imports) and self time (excluding
             nested imports). Note that its output may be broken in multi-threaded
             application. Typical usage is python3 -X importtime -c 'import asyncio'
         -X dev: enable CPython's "development mode", introducing additional runtime
             checks which are too expensive to be enabled by default. Effect of the
             developer mode:
                * Add default warning filter, as -W default
                * Install debug hooks on memory allocators: see the PyMem_SetupDebugHooks()
                  C function
                * Enable the faulthandler module to dump the Python traceback on a crash
                * Enable asyncio debug mode
                * Set the dev_mode attribute of sys.flags to True
                * io.IOBase destructor logs close() exceptions
         -X utf8: enable UTF-8 mode for operating system interfaces, overriding the default
             locale-aware mode. -X utf8=0 explicitly disables UTF-8 mode (even when it would
             otherwise activate automatically)
         -X pycache_prefix=PATH: enable writing .pyc files to a parallel tree rooted at the
             given directory instead of to the code tree
         -X warn_default_encoding: enable opt-in EncodingWarning for 'encoding=None'
```
## Useful ENV variables

```
export OMP_NUM_THREADS=1
export PYTORCH_JIT=1

export CUDA_LAUNCH_BLOCKING=1
```
## Storage Performnace and Data Loading

If the code takes too long to load data from disk (local or network storage), I recommend checking I/O performance before profiling the code thoroughly. Estimating storage performance is a complex task, but the following tool can provide some numbers to start with.

* https://github.com/bkryza/naive-bench

Here are some tutorials and libraries to optimise the data loading part of the code:
* https://blog.genesiscloud.com/2023/tutorial-series-how-to-optimize-IO-performance
* https://github.com/NVIDIA/DALI

