export OMP_NUM_THREADS=8

for i in {1..6}
 do
	 sleep 20
	 $@ seed=$i &
done
