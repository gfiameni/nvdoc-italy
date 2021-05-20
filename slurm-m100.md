### Run one task of myApp on one core of a node:

`$ ssh login.m100.cineca.it -l <your username>`

`$ srun myApp`

This is the simplest way to run a job on a cluster.  In this example, the lone srun command defaults to asking for one task on one core on one node of the default queue charging the default account.

Run hostname in an interactive allocation:

`$ salloc`

```salloc: no partition specified, using default partition m100_all_serial
salloc: Pending job allocation 3037981
salloc: job 3037981 queued and waiting for resources
```

blocks here until job runs

```
salloc: job 3037981 has been allocated resources
salloc: Granted job allocation 3037981
salloc: Waiting for resource configuration
salloc: Nodes login06 are ready for job
manpath: warning: $MANPATH set, ignoring /etc/man_db.conf
```

$ srun hostname
login06

Run it again:

`$ srun -n 4 hostname`
`login06`
`login06`
`login06`
`login06`

Now exit the job and allocation

`$ exit`
`exit`
`salloc: Relinquishing job allocation 3037981`
`salloc: Job allocation 3037981 has been revoked.`

Like srun in the first example, salloc defaults to asking for one node of the default queue charging the default account.  Once the job runs and the prompt appears, any further commands are run within the job's allocated resources until exit is invoked.
Create a batch job script and submit it

$ cat > myBatch.cmd
#!/bin/bash
#SBATCH -N 4
#SBATCH -p pdebug
#SBATCH -A myAccount
#SBATCH -t 30

srun -N 4 -n 32 myApp
^D

This script asks for 4 nodes from the pdebug queue for no more than 30 minutes charging the myAccount account.  The srun command launches 32 tasks of myApp across the four nodes.

Now submit the job:

$ sbatch myBatch.cmd
Submitted batch job 150104

See the job pending in the queue:

$ squeue
  JOBID PARTITION     NAME     USER  ST       TIME  NODES NODELIST(REASON)
 150104    pdebug myBatch.       me  PD       0:00      4 (Priority)

After the job runs, the output will be found in a file named after the job id:  slurm-150104.out
See only your jobs in the queue

$ squeue -u <myName>

See all the jobs in the queue

$ squeue

List queued jobs displaying the fields that are important to you

$ man squeue

and scroll to the output format specifiers listed under the -o option.  Then create an environment variable that contains the fields you like to see.

For example, for bash:

$ export SQUEUE_FORMAT="%.7i %.8u %.8a %.9P %.5D %.2t %.19S %.8M %.10l %.10Q"

and for csh:

$ setenv SQUEUE_FORMAT "%.7i %.8u %.8a %.9P %.5D %.2t %.19S %.8M %.10l %.10Q"

Now run squeue:

$ squeue
  JOBID     USER  ACCOUNT PARTITION NODES ST START_TIME     TIME  TIMELIMIT   PRIORITY
 147445    carol   guests    pbatch    32 PD        N/A     0:00    5:00:00    1005641
 147446    henry   guests    pbatch    32 PD        N/A     0:00    5:00:00    1004454
 147447      sue   guests    pbatch    32 PD        N/A     0:00    5:00:00    1004296

Display the pending jobs ordered by decreasing priority

$ squeue -t pd -S-p

Display details about a specific job

$ scontrol show job <jobID>

Display the job script for one of your jobs

$ scontrol -dd show job <jobID>

Show all the jobs you have run today

$ sacct -X
       JobID    JobName  Partition    Account  AllocCPUS      State ExitCode
------------ ---------- ---------- ---------- ---------- ---------- --------
150096               sh     pbatch         lc         72  COMPLETED      0:0
150104       myBatch.c+     pdebug         lc        288  COMPLETED      0:0
150106       myBatch.c+     pdebug         lc        288  COMPLETED      0:0
150109       myBatch.c+     pdebug         lc        288  COMPLETED      0:0
150112       libyogrtt+     pdebug         lc         72  COMPLETED      0:0
150121       myBatch.c+     pdebug         lc        288  COMPLETED      0:0

Show all the job steps that ran within a specific job

$ sacct -j 150065
       JobID    JobName  Partition    Account  AllocCPUS      State ExitCode
------------ ---------- ---------- ---------- ---------- ---------- --------
150065       myBatch.c+     pdebug         lc         72  COMPLETED      0:0
150065.batch      batch                    lc          1  COMPLETED      0:0
150065.0         my_mpi                    lc          2  COMPLETED      0:0
150065.1         my_mpi                    lc          2  COMPLETED      0:0
150065.2         my_mpi                    lc          2  COMPLETED      0:0
150065.3         my_mpi                    lc          2  COMPLETED      0:0
150065.4         my_mpi                    lc          2  COMPLETED      0:0

List the charge accounts you are permitted to use (sbatch/salloc/srun -A option)

$ sshare

This command also shows the historical usage your jobs have accrued to each charge account.  The fair-share factor is also displayed for you for each of your accounts.  This factor will be used in calculating the priority for your current pending jobs and any job you submit.  For details, see Multi-factor Priority and Fair-Tree.
Display the factors contributing to each pending job's assigned priority

$ sprio -l
  JOBID     USER   PRIORITY     AGE  FAIRSHARE    JOBSIZE  PARTITION        QOS   NICE
 143104  harriet    1293802   28234     265568          0          0    1000000      0
 143105      sam    1293802   28234     265568          0          0    1000000      0

Cancel a job, whether it is pending in the queue or running

$ scancel <job_ID>

Send a signal to a running job

For example, send SIGUSR1:

$ scancel -s USR1 <job_ID>

Display the queues available

$ sinfo

Display details about all the queues (aka partitions)

$ scontrol show partition
