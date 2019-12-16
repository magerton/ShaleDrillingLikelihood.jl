echo $Env:JULIA_NUM_THREADS
$origthreads = $Env:JULIA_NUM_THREADS
# $Env:PATH += ";C:\Users\magerton\AppData\Local\Julia-1.3.0\bin"
foreach ($i in 1,2) {
    $env:JULIA_NUM_THREADS = $i
    
    echo "Julia 1.1.1"
    C:\Users\magerton\AppData\Local\Julia-1.1.1\bin\julia.exe ".\full-model\optimize-dynamic-speedtest.jl"
    
    echo "Julia 1.2.0"
    C:\Users\magerton\AppData\Local\Julia-1.2.0\bin\julia.exe ".\full-model\optimize-dynamic-speedtest.jl"
    
    echo "Julia 1.3.0"
    C:\Users\magerton\AppData\Local\Julia-1.3.0\bin\julia.exe ".\full-model\optimize-dynamic-speedtest.jl"
}
$env:JULIA_NUM_THREADS = $origthreads
