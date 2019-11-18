echo $Env:JULIA_NUM_THREADS
$origthreads = $Env:JULIA_NUM_THREADS
$Env:PATH += ";C:\Users\magerton\AppData\Local\Julia-1.2.0\bin"
for($i = 1; $i -le $origthreads; $i=$i+1){
    $env:JULIA_NUM_THREADS = $i
    julia.exe runtests.jl
}
