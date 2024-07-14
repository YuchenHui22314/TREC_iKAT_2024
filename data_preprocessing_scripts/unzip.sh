for file in *.bz2; do
    bzip2 -d "$file"
done
