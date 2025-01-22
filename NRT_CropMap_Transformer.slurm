#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name="OK"
#SBATCH --output="Log/output.%j.%N.log"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40  # bigmen=80, others=40
#SBATCH --mem 507904M # in unit MB 507904M=496G; is the max of the innovator 
##SBATCH --mem 254004M # in unit MB 507904M=496G; is the 1/2 max of the innovator 
#SBATCH --time=14-00:00:00    #Days-Hours:Minutes:Seconds, e.g. 1-00:00:00 represents 1 day
#SBATCH --partition=gpu       # compute(54 40cors 120gb mem), bigmem(5-80cors,3tb mem), gpu(4, 40cores, 798gb mem)
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL #see https://slurm.schedmd.com/sbatch.html for all options.
#SBATCH --mail-user=hankui.zhang@sdstate.edu

# =========== implementation ================================================================= #
# cd /mmfs1/home/jacks.local/hankui.zhang/mycode.research/LCMAP/HLS_crop_type/classification2
# sbatch NRT_CropMap_Transformer.slurm
# ============================================================================================ #

## ============= load modules ================================================================ #
module purge
module load python/3.8.15
module load rasterio/tensorflow
module load gdal
# module load tensorflow/2.14.0
# module load rasterio/1.3.9

# source activate khuong
# module load rasterio
# module load R
# module load perl

# gdal
# module load gdal/3.3.3
# module load geos

# hdf
# module load hdf4/4.2.14-no-netcdf
# module load hdf5/1.10.4-mpich-3.3-szip-2.1.1
# module load hdfeos2
# module load hdfeos5/1.16

# 
date
## ============= [1] Single Job============================================================== #
# set command: python, Rcript, perl
cmd=python

# set the dir of input script
indir=/location_of_Pro_load_model_run_tile_v5_0.py

# set script file names (space separate)
file=("Pro_load_model_run_tile_v5_1.py")

# for file in ${files[@]}; do
$cmd $indir/$file /mmfs1/scratch/jacks.local/yu.shen2069/Data/HLS/Raw/ /mmfs1/scratch/jacks.local/hankui.zhang/workspace/hls_transformer/test_output/ T14TLT 2023 190

# /mmfs1/scratch/jacks.local/yu.shen2069/Data/HLS/Raw /mmfs1/scratch/jacks.local/hankui.zhang/workspace/hls_transformer/test_output T14TLT 2023 190

## ============= Single Job================================================================== # 
date