# Default target
all: run_large extract

# Run experiments for large graph
run_large:
	@echo "Running experiments for large graph..."
	@rm -f performance.log
	@touch performance.log
	@echo "MPI=2, Threads=2, Batch=1000, Async=2"
	@OMP_NUM_THREADS=2 mpirun -np 2 ./sssp_update -g epinions.metis -c changes_large.txt -s 1 -o sssp_result.txt -l performance.log -b 1000 -a 2 > /dev/null
	@echo "MPI=4, Threads=2, Batch=1000, Async=2"
	@OMP_NUM_THREADS=2 mpirun -np 4 ./sssp_update -g epinions.metis -c changes_large.txt -s 1 -o sssp_result.txt -l performance.log -b 1000 -a 2 > /dev/null
	@echo "MPI=4, Threads=4, Batch=1000, Async=2"
	@OMP_NUM_THREADS=4 mpirun -np 4 ./sssp_update -g epinions.metis -c changes_large.txt -s 1 -o sssp_result.txt -l performance.log -b 1000 -a 2 > /dev/null
	@echo "MPI=4, Threads=8, Batch=1000, Async=2"
	@OMP_NUM_THREADS=8 mpirun -np 4 ./sssp_update -g epinions.metis -c changes_large.txt -s 1 -o sssp_result.txt -l performance.log -b 1000 -a 2 > /dev/null
	@echo "MPI=4, Threads=4, Batch=500, Async=2"
	@OMP_NUM_THREADS=4 mpirun -np 4 ./sssp_update -g epinions.metis -c changes_large.txt -s 1 -o sssp_result.txt -l performance.log -b 500 -a 2 > /dev/null
	@echo "MPI=4, Threads=4, Batch=2000, Async=2"
	@OMP_NUM_THREADS=4 mpirun -np 4 ./sssp_update -g epinions.metis -c changes_large.txt -s 1 -o sssp_result.txt -l performance.log -b 2000 -a 2 > /dev/null
	@echo "MPI=4, Threads=4, Batch=1000, Async=1"
	@OMP_NUM_THREADS=4 mpirun -np 4 ./sssp_update -g epinions.metis -c changes_large.txt -s 1 -o sssp_result.txt -l performance.log -b 1000 -a 1 > /dev/null
	@echo "MPI=4, Threads=4, Batch=1000, Async=4"
	@OMP_NUM_THREADS=4 mpirun -np 4 ./sssp_update -g epinions.metis -c changes_large.txt -s 1 -o sssp_result.txt -l performance.log -b 1000 -a 4 > /dev/null

# Extract performance data
extract:
	@echo "Extracting performance data..."
	@echo "Dataset,MPI Processes,OpenMP Threads,Batch Size,Async Level,Vertices,Edges,Changes,Initial SSSP Time (ms),Update SSSP Time (ms)" > results.txt
	@grep -A 8 "MPI Processes:" performance.log | \
	awk 'BEGIN { dataset="Large" } \
		/MPI Processes:/ { mpi=$$3 } \
		/OpenMP Threads:/ { threads=$$3 } \
		/Batch Size:/ { batch=$$3 } \
		/Async Level:/ { async=$$3 } \
		/Vertices:/ { vertices=$$2 } \
		/Edges:/ { edges=$$2 } \
		/Changes:/ { changes=$$2 } \
		/Initial SSSP Time \(ms\):/ { initial=$$5 } \
		/Update SSSP Time \(ms\):/ { update=$$5; print dataset "," mpi "," threads "," batch "," async "," vertices "," edges "," changes "," initial "," update }' \
		>> results.txt

# Clean up
clean:
	rm -f performance.log results.txt sssp_result.txt

.PHONY: all run_large extract clean
