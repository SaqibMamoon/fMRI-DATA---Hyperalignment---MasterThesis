

abc=$(3dinfo 100307_clusterorder+tlrc. | grep datum | awk '{print $12}')

for i in $(seq -w 1 $abc); 
do 

 -prefix cluster_$i


echo $i; 

done
