#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
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
#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
.KEEP_STATE:

all: seq_main seq_main_1 seq_main_2

# Compiler and flags
CC          = gcc
INCFLAGS    = -I.
OPTFLAGS    = -O4 -DNDEBUG -g
LDFLAGS     = $(OPTFLAGS) -fopenmp  # Ensure OpenMP linking here
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS) -fopenmp  # Add OpenMP for compilation too

# Header files and common source files
H_FILES     = kmeans.h
COMM_SRC    = file_io.c util.c
COMMON_OBJ  = $(COMM_SRC:%.c=%.o) wtime.o

# Source files for each executable
SEQ_SRC     = printless_main.c seq_kmeans.c
SEQ1_SRC    = printless_main.c seq_kmeans_1.c
SEQ2_SRC    = printless_main.c seq_kmeans_2.c

# Object files for each executable
SEQ_OBJ     = $(SEQ_SRC:%.c=%.o)
SEQ1_OBJ    = $(SEQ1_SRC:%.c=%.o)
SEQ2_OBJ    = $(SEQ2_SRC:%.c=%.o)

# Rules to build object files
$(SEQ_OBJ) $(SEQ1_OBJ) $(SEQ2_OBJ): $(H_FILES)

seq_main.o: printless_main.c $(H_FILES)
	$(CC) $(CFLAGS) -c $*.c

seq_kmeans.o: seq_kmeans.c $(H_FILES)
	$(CC) $(CFLAGS) -c $*.c

seq_kmeans_1.o: seq_kmeans_1.c $(H_FILES)
	$(CC) $(CFLAGS) -c $*.c

seq_kmeans_2.o: seq_kmeans_2.c $(H_FILES)
	$(CC) $(CFLAGS) -c $*.c

wtime.o: wtime.c
	$(CC) $(CFLAGS) -c $*.c

# Executable targets
seq_main: $(SEQ_OBJ) $(COMMON_OBJ)
	$(CC) $(LDFLAGS) -o $@ $(SEQ_OBJ) $(COMMON_OBJ) $(LIBS)

seq_main_1: $(SEQ1_OBJ) $(COMMON_OBJ)
	$(CC) $(LDFLAGS) -o $@ $(SEQ1_OBJ) $(COMMON_OBJ) $(LIBS)

seq_main_2: $(SEQ2_OBJ) $(COMMON_OBJ)
	$(CC) $(LDFLAGS) -o $@ $(SEQ2_OBJ) $(COMMON_OBJ) $(LIBS)


dist:
	dist_dir=parallel-kmeans \
	&& rm -rf $$dist_dir $$dist_dir.tar.gz\
	&& mkdir -p $$dist_dir/Image_data \
	&& cp $(PACKING_LIST) $$dist_dir \
	&& cp $(INPUTS) $$dist_dir/Image_data \
	&& tar -cf - $$dist_dir | gzip > $$dist_dir.tar.gz \
	&& rm -rf $$dist_dir

clean:
	rm -rf *.o seq_main seq_main_1 seq_main_2 \ \
		core* .make.state              \
		*.cluster_centres *.membership \
		*.cluster_centres.nc *.membership.nc \
		Image_data/*.cluster_centres Image_data/*.membership \
		Image_data/*.cluster_centres.nc Image_data/*.membership.nc

check: all
	# sequential K-means ---------------------------------------------------
	./seq_main -q -b -n 4 -i Image_data/color17695.bin
	./seq_main -q    -n 4 -i Image_data/color100.txt
