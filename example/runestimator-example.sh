#!/bin/bash -l
# NOTE the -l flag!
#    see https://stackoverflow.com/questions/20499596/bash-shebang-option-l

#-----------------------
# Job info
#-----------------------

#SBATCH --job-name=run_estimator
#SBATCH --mail-user=mjagerton@ucdavis.edu
#SBATCH --mail-type=ALL

##SBATCH --output out-%j.output
##SBATCH --error out-%j.output

#-----------------------
# Resource allocation
#-----------------------

#SBATCH --time=4-04:00:00     # in d-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --partition=high2
#SBATCH --mem=256000      # max out RAM

#-----------------------
# script
#-----------------------

echo ""
hostname
module load julia/1.3.0

# create directory for the month
echo ""
MON=$(date +"%Y-%m")
[ -d $MON ] &&  echo "Directory ${MON} exists" || mkdir $MON

# switch into directory
mkdir ${MON}/${SLURM_JOB_ID}
cd    ${MON}/${SLURM_JOB_ID}
echo ""
echo "Starting job!!! ${SLURM_JOB_ID} on partition ${SLURM_JOB_PARTITION}"
echo ""
# print out versions of repos
echo "ShaleDrillingLikelihood commit " $(git -C ~/dev-pkgs/ShaleDrillingLikelihood/ rev-parse HEAD)
echo "haynesville             commit " $(git -C ~/haynesville/ rev-parse HEAD)
echo ""
# print out environment variables
julia -e '[println((k,ENV[k],)) for k in keys(ENV) if occursin("SLURM_NTASKS",k)];'

# run the script
julia --optimize=3 \
    ~/.julia/dev/ShaleDrillingLikelihood/example/run-estimator.jl \
    --dataset='data_last_lease_only.RData' \
    --cost='DrillingCost_TimeFE(2008,2012)' \
    --revenue='DrillingRevenue(Unconstrained(), TimeTrend(), GathProcess() )' \
    --noFull \
    --theta='[-0x1.8fa7df8b8e166p+3, -0x1.1ee5269bb0ed3p+3, -0x1.ec8b9d2469105p+2, -0x1.c85d56d05f875p+2, -0x1.b210f34d44434p+2, 0x1.9385b5338096bp+0, -0x1.7ed2cc582d0ecp+0, -0x1.5abeccb6fa743p+1, 0x1.319f90114062cp-1, 0x1.5c011ab314e82p-2, 0x1.6d3b588f33071p-6, 0x1.5cdb42f5ea573p-1, 0x1.cf6a2c88c90bcp-4, 0x1.32ba584838fe7p-1, 0x1.2eb9aed1b3c82p+0, -0x1.b2a0476a1b014p+0, 0x1.1ea18d15d9801p-3, 0x1.ef0a7dbd79c02p+1, 0x1.0cf72d7b30f84p+2, 0x1.4391e744b338bp+2, 0x1.7d20c6a7c4927p+2, 0x1.a1efd4a1176a5p+2, -0x1.d8fbbf907b493p+3, 0x1.8c726cf39a8d3p-4, 0x1.477d354182cfdp-2, ]' \
    --Mcnstr=500 \
    --Mfull=2000 \
    --maxtimeCnstr=3*60^2 \
    --maxtimeFull=48*60^2 \
    --numP=51 \
    --numPsi=51 \
    --extendPriceGrid='log(3)' \
    --minTransProb=1e-5
