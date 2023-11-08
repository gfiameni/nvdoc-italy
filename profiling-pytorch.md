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

NVIDIA Extend Toolkit is can be used to profile the NVIDIA GPU 

NVTX cannot use on its own. We need to feed the NVTX result into some visualization software/tools
such as NSight System

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

cProfile is a build-in profiler at python specially for CPU event profiling. There are different ways to use cProfiler.

Profile Metrics:
  - CPU
    - ncalls: The number of calls made.
    - tottime: The total time spent in the given function (excluding time made in calls to sub-functions), expressed in seconds.
    - percall: The time spent per call, calculated as tottime/ncalls.
    - cumtime: The cumulative time spent in this and all subfunctions.
    - percall: The cumulative time per primitive call (meaning calls that aren’t via recursion), calculated as cumtime/primitive calls.
Pros: 
  - Easy to use
Cons:
  - Unable to identify the script lines with most resources spent 
  - low level subfunction call profiling
  - Only CPU profiling provided
  - Less customization

### Method 1
1. Import the script you want to profile
```
import self_defined_script
import cProfile
```
2. Directly call cProfile.run()
```
cProfile.run(self_defined_script.funct_to_profile())
```
You can create an entry point like run() or main() in the script for profiling

### Method 2
1. Define the profiler instance
```
pr = cProfile.Profile()
```
2. Start the profiler. cProfile will start to collect CPU data at this point
```
pr.enable()
```
3. Then add the codes need to be profiled
4. Stop the profiler. 
```
pr.disable()
```
You can get the result in the script by `pr.print_stats()` or you can dump the result into external file by `pr.dump_stats('output.prof')`

An Example of `pr.print_stats()`

![image](https://github.com/gfiameni/nvdoc-italy/assets/57800717/3501ce47-b84f-4a94-a40f-66c8ff422864)

Sort the profiling result before print `pr.sort_stats('cumulative').print_stats()`


## PyTorch Profiler
* https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

Pytorch profiler is Integrated high-performance CPU, GPU, and memory profiler

Profile Metrics:
  - CPU:
    - Self CPU%: The percentage of total CPU time spent in the function excluding its children functions.
    - Self CPU: The time that the CPU spent in the function excluding its children functions.
    - CPU Total %: The percentage of total CPU time spent in the function including its children functions.
    - CPU Total: The total time that the CPU spent in the function including its children functions.
    - CPU Time Avg: The average time the CPU spent in the function per call.
  - GPU (CUDA):
    - Self CUDA: The time that the GPU spent in the function excluding its children functions.
    - Self CUDA%: The percentage of total GPU time spent in the function excluding its children functions.
    - CUDA Total: The total time that the GPU spent in the function including its children functions.
    - CUDA Time Avg: The average time the GPU spent in the function per call.
  - Memory
    - CPU Mem: The amount of CPU memory used by the function including its children functions.
    - Self CPU Mem: The amount of CPU memory used specifically by the function, excluding its children functions.
    - CUDA Mem: The amount of GPU memory used by the function including its children functions.
    - Self CUDA Mem: The amount of GPU memory used specifically by the function, excluding its children functions.
  - Execution
    - \# of Calls: The number of times the function was called.
    - Total GFLOPS: The total number of billion Floating Point Operations (FLOPs) that the function performed. This is a measure of computational intensity and can be useful for understanding the computational cost of the function.
- Pros: 
  - Easy to use
  - Good customization
  - Support multiple hardware profiling
- Cons:
  - Unable to identify the script lines with most resources spent 
  - low level subfunction call profiling

To use the pytorch_profiler
```
# 1. Start profiling setting
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,    # Monitor CPU activities
        torch.profiler.ProfilerActivity.CUDA,   # Monitor GPU CUDA activities
    ],
    record_shapes=True,  # Record the shape of the input
    profile_memory=True, # Monitor the memory usage
    with_stack=True,     # Record source information for the line
    with_flops=True,     # Estimate the FLOPS (Floating point operation) with formula
    use_cuda=True,       # Measure CUDA kernel execution time
    with_modules=False,  # Record module hierarchy int the callstack
) as prof:
# End profiling setting

  # Your code here
  
# Print out the profiling result
print(prof.key_averages())

# Print out profiling result based on cpu time and limit only 10 rows.
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Record the profiling results
prof.export_chrome_trace("pytorch_profiler_trace.json")
print('Chrome trace saved!')

prof.export_stacks("/tmp/torch_cuda_stack.txt", "self_cuda_time_total")
print('Cuda stack saved!')

prof.export_stacks("/tmp/torch_cpu_stack.txt", metric="self_cpu_time_total")
print('CPU stack saved!')
```

An example of pytorch_profiler output

![image](https://github.com/gfiameni/nvdoc-italy/assets/57800717/f55386d4-2f89-4a83-b267-36cf84f084f9)

User can also add record function to wrap the lines together into single profile
```
with torch.profiler.profile(with_stack=True, profile_memory=True) as prof:
    with torch.profiler.record_function("my_operation"):
        # Your code here
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
```

An example of using profiler with record function
![image](https://github.com/gfiameni/nvdoc-italy/assets/57800717/c727daec-288f-4c7a-b8c2-b4c10a107e74)

## PyTorch Lightning Profiler
* https://pytorch-lightning.readthedocs.io/en/1.5.10/advanced/profiler.html

PyTorch Lightning is a lightweight PyTorch wrapper for high-performance AI research. It provides a high-level interface for PyTorch and reduces the boilerplate code.

PyTorch Lightning includes a built-in Profiler class that provides simple tools to profile your Lightning module and Lightning callbacks. You can use it to identify bottlenecks in your models/training loops.

The Lightning Profiler is different from other profiling tools because it's specifically designed to work with PyTorch Lightning and it's integrated directly into the Lightning training loop.

- Profile Metrics:
  - CPU
    - CPU time
    - ncalls: The number of calls made.
    - tottime: The total time spent in the given function (excluding time made in calls to sub-functions), expressed in seconds.
    - percall: The time spent per call, calculated as tottime/ncalls.
    - cumtime: The cumulative time spent in this and all subfunctions.
    - percall: The cumulative time per primitive call (meaning calls that aren’t via recursion), calculated as cumtime/primitive calls.
  - GPU
    - GPU time 
  - Pipeline
    - Number of line calls
  
- Pros:
  - Details and sophisticated profiling
- Cons: 
  - Only works on Pytorch Lightning Frameworks' code meaning it can only profile the ML models and pipeline with pytorch lightning only.

### Simple profiler
Simple profiler only profile the CPU time and mean CPU time for each operations, GPU, memory and IO count is not included. 

To use the simple version of Pytorch Lightning profiler, user need to define a Lightning model class for profiling. 

Below show a sample of skeleton code of Bert classfication model

``` sample
class BertClassifier(pl.LightningModule):
    def __init__(self, learning_rate=2e-5):
        super().__init__()
        self.learning_rate = learning_rate
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dataset = load_dataset('imdb')

    def forward(self, input_ids, attention_mask, labels=None):
        ...

    def training_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        ...

    def tokenize_function(self, examples):
        ...

    def prepare_data(self):
        ...
```

Next, create the trainer instance and train with the Pytorch Lighning. Then fit the model and the profiler will automatically profile the results.


```
trainer = pl.Trainer(max_epochs=3,profiler="simple")
model = BertClassifier()
trainer.fit(model)
print(trainer.profiler.summary())
```


- Sample
  
![image](https://github.com/gfiameni/nvdoc-italy/assets/57800717/6033c484-7fcf-4869-945f-59a6b13fe218)

### Advance profiler

Pytorch Lightning Profiler also support a more deatailed profiling by using cProfile

Just need to set modify the trainer parameter like `pl.Trainer(max_epochs=3, profiler="Advanced")`

- Sample

![image](https://github.com/gfiameni/nvdoc-italy/assets/57800717/b7e94a6b-e110-4869-ac18-23fd5a4fc36f)



## Scalene
* https://github.com/emeryberger/scalene

Offical profiler comparison graph 
ref: https://github.com/emeryberger/scalene#scalene-a-python-cpugpumemory-profiler-with-ai-powered-optimization-proposals
![image](https://github.com/gfiameni/nvdoc-italy/assets/57800717/364b14a2-c288-41db-860d-9e00b8bc672f)

- Profile Metrics:
  - CPU
    - CPU Time: Scalene measures the time spent in each Python function, providing a high-resolution, low-overhead view of CPU usage.
  - Memory
    - Memory Usage: Scalene also profiles the memory consumption of your Python program. It shows the amount of memory used on a line-by-line basis, helping you identify memory-intensive parts of your code.
  - GPU:
    - GPU Time: If you're using PyTorch, Scalene can measure the time spent on GPU computations. This can help you optimize your code for GPU usage.
  - I/O:
    - Copying Activity: Scalene can track and report the amount of data your program copies between Python and native code. This can help you identify performance bottlenecks related to data movement.
  - Execution:
    - Number of Function Calls: Scalene reports the number of times each function is called.
    - Percentage of Time: For each line of code, Scalene provides the percentage of time spent executing that line relative to the total run time.
- Pros:
  - Do not require to modify the code specificlly
  - Good customization
  - Good functionality
  - Support multiple hardware profiling
  - Support GUI
- Cons:
  - Unable to identify the script lines with most resources spent 
  - low level subfunction call profiling


## Python Line Profiler
* https://github.com/pyutils/line_profiler

Line Profiler is a python line by line profiling tools that used to records the execution time used

- Profile Metrics:
  - Executions:
    - Line #: Specifies the line number in the script.
    - Line Contents: The exact source code from the line. 
    - Hits: Represents the count of executions for a particular line.
    - Per Hit: Average execution time for a single hit, expressed in the timer's units.
    - Time: Total time spent executing the line, given in the timer's units (conversion factor to seconds is provided in the header).
    - % Time: Percentage of the total execution time of the function spent on this line.
  - Pros:
    - Easy to use
    - Direct and simple result
    - High level lines and function call profiling
  - Cons:
    - Less customizations
    - Not designed to use at inline

To use the line profiler

```
from line_profiler import profile

# add @profile to the function that needs profiling

@profile
def model_train(model, epoch, train_loader, val_loader, optimizer, device, record_result:bool=False):
    # yours code here
```

kernprof is used to run line profiler

below show the official `kernprof --help`
```
usage: kernprof [-h] [-V] [-l] [-b] [-o OUTFILE] [-s SETUP] [-v] [-r] [-u UNIT] [-z]
                [-i [OUTPUT_INTERVAL]] [-p PROF_MOD] [--prof-imports]
                script ...

Run and profile a python script.

positional arguments:
  script                The python script file to run
  args                  Optional script arguments

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  -l, --line-by-line    Use the line-by-line profiler instead of cProfile. Implies --builtin.
  -b, --builtin         Put 'profile' in the builtins. Use 'profile.enable()'/'.disable()',
                        '@profile' to decorate functions, or 'with profile:' to profile a section
                        of code.
  -o OUTFILE, --outfile OUTFILE
                        Save stats to <outfile> (default: 'scriptname.lprof' with --line-by-line,
                        'scriptname.prof' without)
  -s SETUP, --setup SETUP
                        Code to execute before the code to profile
  -v, --view            View the results of the profile in addition to saving it
  -r, --rich            Use rich formatting if viewing output
  -u UNIT, --unit UNIT  Output unit (in seconds) in which the timing info is displayed (default:
                        1e-6)
  -z, --skip-zero       Hide functions which have not been called
  -i [OUTPUT_INTERVAL], --output-interval [OUTPUT_INTERVAL]
                        Enables outputting of cumulative profiling results to file every n
                        seconds. Uses the threading module. Minimum value is 1 (second). Defaults
                        to disabled.
  -p PROF_MOD, --prof-mod PROF_MOD
                        List of modules, functions and/or classes to profile specified by their
                        name or path. List is comma separated, adding the current script path
                        profiles full script. Only works with line_profiler -l, --line-by-line
  --prof-imports        If specified, modules specified to `--prof-mod` will also autoprofile
                        modules that they import. Only works with line_profiler -l, --line-by-line
```

There are 2 methods to profile with line profiler

## Method 1

Open a terminal to run the profiling command
```
kernprof -l -v script_to_profile.py
```
kernprof will a binary file `script_to_profile.py.lprof`

To view the result, we need to use the line_profiler to parse the result into human readable format 

`python -m line_profiler script_to_profile.py.lprof`

## Method 2

We can call the profiler inline inside the python function

```
%load_ext line_profiler
%lprun -f funct_for_profile funct_for_profile(*args)
```

Here show an example to profile function `funct1(data, arg1, arg2)`
```
%load_ext line_profiler
%lprun -f funct1 funct1(data, arg1, arg2)
```

if the funct_for_profile will return object, this is recommended to use method 1 instead as Method 2 cannot correctly handle the output of the funct_for_profile

Example of line_profiler result

![image](https://github.com/gfiameni/nvdoc-italy/assets/57800717/dbdb8a32-86d2-4138-9d5a-3571ed752f89)

## Using with NVIDIA NSight System 

```
# Copy paste the desired command and run it for your app. It will produce a .qdrep file.
# Run the "nsight-sys" GUI executable and File->Open the .qdrep file.
# If you're making the profile locally on your desktop, you may not need nsys at all, you can do
# the whole workflow (create and view profile) through the GUI, but if your job runs remotely on
# a cluster node, I prefer to create .qdrep profiles with nsys remotely, copy them back to my desktop,
# then open them in nsight-sys.
```
## memory-profiler

reference: https://github.com/pythonprofilers/memory_profiler

Memory Profile is a Python module for monitoring memory consumption of a process as well as line-by-line analysis of
 memory consumption for python programs. It's a pure python module which depends on the psutil module.

- Profile Metrics:
  - Memory
    - line by line memory heap increasement after executing the line
    - Total memory usage
    - line occurance
- Pros:
  - Easy and direct to use
  - line by line profile
  - easy to cehck memory leasks
- Cons:
  - Only simple memory usage recorded, detailed variables' name and space allocation is not provided.

To use the memory profiler is very direct 

Import the profil from the package
'from memory_profiler import profile'

Add the `@prfile` decorator into the functions need profile.
Note that it will not profile any subfunctions.
```
@profile
def my_func():
    #  yours code here
    return a
```

Then we can directly use any IDE or CLI to run the script.

for CLI:
we may use `python -m memory_profiler example.py` to run the memory profiler.

Below show the example. 
![image](https://github.com/gfiameni/nvdoc-italy/assets/57800717/f2b8ff73-5691-4791-9ff7-da0ec0233811)


 

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
## Storage Performance and Data Loading

If the code takes too long to load data from disk (local or network storage), I recommend checking I/O performance before profiling the code thoroughly. Estimating storage performance is a complex task, but the following tool can provide some numbers to start with.

* https://github.com/bkryza/naive-bench

Here are some tutorials and libraries to optimise the data loading part of the code:
* https://blog.genesiscloud.com/2023/tutorial-series-how-to-optimize-IO-performance
* https://github.com/NVIDIA/DALI

