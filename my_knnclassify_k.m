function [ cluster_confidence_matrix1 ] = my_knnclassify_k( test_set,training_set,group,k,no_of_clusters )




cluster_confidence_matrix1=zeros(length(test_set),no_of_clusters);



for row=1:1:size(test_set,1)

clear d;
d=zeros(1,size(training_set,1));

for j=1:1:size(training_set,1)    
d(1,j)=norm(test_set(row,:)-training_set(j,:));
end


if isequal(training_set,test_set)
    d(1,row)=9999;
end

[d_sorted, I] = sort(d);
val = d_sorted(:, 1:k);
idx = I(:, 1:k);


for my_idx=1:1:k
for z=1:1:no_of_clusters
    if group(idx(my_idx),1)==z
        %idx(my_idx)
        cluster_confidence_matrix1(row,z)=cluster_confidence_matrix1(row,z)+1;
    end
end


end

cluster_confidence_matrix1(row,:)=cluster_confidence_matrix1(row,:)/sum(cluster_confidence_matrix1(row,:));

end
 


end

