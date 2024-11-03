#!/bin/bash


input_files=("color17695.bin" "edge17695.bin" "texture17695.bin")
correctResults=0

cd opt_src
make 

# Run the seq_means program with specified arguments for all files from input files
# Store the output as golden_output
for i in "${input_files[@]}"
do
    ./seq_main -o -b -n 2000 -i ../Image_data/$i
    cp ../Image_data/$i.membership golden_$i.membership
done

# Loop over each parallel version and each input file
for exe in par_main_{1..8}
do
    for i in "${input_files[@]}"
    do
        ./$exe -o -b -n 2000 -i ../Image_data/$i

        # Compare the output of the program with the golden output
        diff ../Image_data/$i.membership golden_$i.membership > diff_output.txt

        if [ -s diff_output.txt ]
        then
            echo "Output of $exe is not correct for $i"
        else
            correctResults=$((correctResults+1))
        fi
    done
done

echo "Number of correct results: $correctResults/$(( ${#input_files[@]} * 8))"

make clean

rm diff_output.txt
