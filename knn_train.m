
for k=1:1:15
clear group;
group=training_data(:,size(test_data,2));
cluster_confidence_matrix_temp1 = my_knnclassify_k(training_data(:,1:size(training_data,2)-1),training_data(:,1:size(training_data,2)-1),group,k,no_of_clusters);
clear sum;
train_cluster_obtained1=zeros(length(cluster_confidence_matrix_temp1),1);
for row=1:1:length(training_data)
train_cluster_obtained1(row,1)=find(cluster_confidence_matrix_temp1(row,:)==max(cluster_confidence_matrix_temp1(row,:)),1);
end
cluster_confidence_matrix_temp1
correct=0;
chk=[cluster_vector train_cluster_obtained1];
for i=1:1:length(chk)
    if chk(i,1)==chk(i,2)
        correct=correct+1;
    end
end
%cluster_confidence_matrix_temp1
accuracy(k,1)=correct*100/length(chk);
end
accuracy
sum(accuracy)/15
k=find(accuracy==max(accuracy),1)
accuracy(k,1)
cluster_confidence_matrix_temp1 = my_knnclassify_k(training_data(:,1:size(training_data,2)-1),training_data(:,1:size(training_data,2)-1),group,k,no_of_clusters);
