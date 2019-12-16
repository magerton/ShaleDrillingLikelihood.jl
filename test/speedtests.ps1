echo $Env:JULIA_NUM_THREADS
$origthreads = $Env:JULIA_NUM_THREADS
$Env:PATH += ";C:\Users\magerton\AppData\Local\Julia-1.2.0\bin"
foreach ($i in 1,2,4,8) {
    $env:JULIA_NUM_THREADS = $i
    julia.exe ".\full-model\optimize-dynamic-speedtest.jl"
}
