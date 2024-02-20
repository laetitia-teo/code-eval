for script in run_jz_llama3bs*.sh; do
    sbatch "$script"
done
