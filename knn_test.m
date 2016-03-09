


clear group;
group=training_data(:,size(training_data,2));
obtained_cluster_confidence_vector1_temp = my_knnclassify_k(test_data(:,1:size(test_data,2)-1),training_data(:,1:size(training_data,2)-1),group,k,no_of_clusters);
clear sum
test_cluster_obtained1=zeros(length(obtained_cluster_confidence_vector1_temp),1);
for row=1:1:length(test_data)
    test_cluster_obtained1(row,1)=find(obtained_cluster_confidence_vector1_temp(row,:)==max(obtained_cluster_confidence_vector1_temp(row,:)),1);
end

obtained_cluster_confidence_vector1_temp


%extra_col1=zeros(size(obtained_cluster_confidence_vector1_temp,1),size(obtained_cluster_confidence_vector1_temp,2)+1);
%extra_col1(:,1:size(obtained_cluster_confidence_vector1_temp,2))=obtained_cluster_confidence_vector1_temp;
%prob_correct=0;
%clear doub
%doub=[test_cluster_obtained1 test_data(:,size(test_data,2))];
%for i=1:1:size(doub,1)
%for j=1:1:no_of_classes
%if doub(i,1)==doub(i,2)
%if doub(i,2)==j&&((doub(i,1)==2*j)||(doub(i,1)==2*j-1))
%    prob_correct=prob_correct+1;
%   extra_col1(i,size(obtained_cluster_confidence_vector1_temp,2)+1)=1;
%end
%end
%end
%extra_col1;
%probable_accuracy=prob_correct*100/size(doub,1)
