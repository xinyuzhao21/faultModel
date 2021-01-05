for f in experiments/*.sh; do  # or wget-*.sh instead of *.sh
  sbatch "$f"
done