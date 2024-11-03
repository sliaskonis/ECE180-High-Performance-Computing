# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    File:         Makefile                                                  */
#    Description:  Makefile for programs running a simple k-means clustering */
#                  algorithm                                                 */
#                                                                            */
#    Author:  Wei-keng Liao                                                  */
#             ECE Department Northwestern University                         */
#             email: wkliao@ece.northwestern.edu                             */
#                                                                            */
#    Copyright (C) 2005, Northwestern University                             */
#    See COPYRIGHT notice in top-level directory.                            */
#                                                                            */
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

.KEEP_STATE:

# Compiler and flags
CC          = gcc
INCFLAGS    = -I.
OPTFLAGS    = -O4 -DNDEBUG -g
CFLAGS      = $(OPTFLAGS) $(INCFLAGS) -fopenmp
LDFLAGS     = $(OPTFLAGS) -fopenmp

# Header files and common source files
H_FILES     = kmeans.h
COMMON_SRC  = file_io.c util.c wtime.c
COMMON_OBJ  = $(COMMON_SRC:.c=.o)

# Source files for each executable
SEQ_SRC     = printless_main.c seq_kmeans.c
PAR_SRC     = parallel_kmeans_1.c parallel_kmeans_2.c parallel_kmeans_3.c parallel_kmeans_4.c parallel_kmeans_5 parallel_kmeans_6.c parallel_kmeans_7.c parallel_kmeans_8.c

# Executable names
EXECS       = seq_main par_main_1 par_main_2 par_main_3 par_main_4 par_main_5 par_main_6 par_main_7 par_main_8

all: $(EXECS)

# Compile object files for each source file
%.o: %.c $(H_FILES)
	$(CC) $(CFLAGS) -c $< -o $@

# Build executable targets
seq_main: $(SEQ_SRC:.c=.o) $(COMMON_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

par_main_%: printless_main.o parallel_kmeans_%.o $(COMMON_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

dist:
	dist_dir=parallel-kmeans \
	&& rm -rf $$dist_dir $$dist_dir.tar.gz \
	&& mkdir -p $$dist_dir/Image_data \
	&& cp $(PACKING_LIST) $$dist_dir \
	&& cp $(INPUTS) $$dist_dir/Image_data \
	&& tar -cf - $$dist_dir | gzip > $$dist_dir.tar.gz \
	&& rm -rf $$dist_dir

clean:
	rm -rf *.o $(EXECS) core* .make.state \
		*.cluster_centres *.membership \
		*.cluster_centres.nc *.membership.nc \
		Image_data/*.cluster_centres Image_data/*.membership \
		Image_data/*.cluster_centres.nc Image_data/*.membership.nc

check: all
	# Sequential K-means
	./seq_main -q -b -n 4 -i ../Image_data/color17695.bin
	./seq_main -q -n 4 -i ../Image_data/color100.txt
