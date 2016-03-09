function [ my_cluster, cluster_count, denominator2 ] = my_cluster_update1( my_cluster,cluster_count, data_item, cluster_no )

size_y=length(data_item);
%cluster_no=data_item(1,size_y);

my_cluster(cluster_no,1:size_y-1)=(cluster_count(1,cluster_no)*my_cluster(cluster_no,1:size_y-1)-data_item(1,1:size_y-1))/(cluster_count(1,cluster_no)-1);
cluster_count(1,cluster_no)=cluster_count(1,cluster_no)-1;

denominator2=max(my_cluster(:,1:size_y-1))-min(my_cluster(:,1:size_y-1));
end

