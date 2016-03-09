function [ my_cluster, cluster_count, denominator2 ] = my_cluster_update3( my_cluster,cluster_count, data_item, new_cluster )

size_y=length(data_item);


my_cluster(new_cluster,1:size_y-1)=(cluster_count(1,new_cluster)*my_cluster(new_cluster,1:size_y-1)+data_item(1,1:size_y-1))/(cluster_count(1,new_cluster)+1);
cluster_count(1,new_cluster)=cluster_count(1,new_cluster)+1;

denominator2=max(my_cluster(:,1:size_y-1))-min(my_cluster(:,1:size_y-1));
end

