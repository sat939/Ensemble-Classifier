
clear
accuracies_for_various_runs=0;
clearvars -except accuracies_for_various_runs
clc
ordered_data=load('iris.txt');
%load cancer_dataset;
%x=cancerInputs';
%y=cancerTargets';
% for i=1:1:size(y,1)
%if y(i,1)==1
%z(i,1)=1;
%else
%z(i,1)=2;
%end
%end
%ordered_data=zeros(size(x,1),size(x,2)+1);
%ordered_data(:,1:size(x,2))=x;
%ordered_data(:,size(x,2)+1)=z;
fraction=0.9;
%class_label=input('enter the class label..\n 5 for iris dataset and 14 for wine\n');
class_label=5;
%no_of_clusters=input('enter no. of clusters ..\n 3 for wine and 9 for iris is preferable to get good clustering\n');
no_of_clusters=6;
random_data=ordered_data(randperm(size(ordered_data,1)),:);
data=random_data;
no_of_classes=length(unique(data(:,class_label)));
[size_x size_y]=size(data); 




training_data=data(1:floor(fraction*size_x),:);
test_data=data(floor(fraction*size_x)+1:size_x,:);

for j=1:1:size(training_data,2)-1
min_data(j)=min(training_data(:,j));
max_data(j)=max(training_data(:,j));
end
%normalisation
for i=1:1:size(training_data,1)
for j=1:1:size_y-1
    training_data(i,j)=(training_data(i,j)-min_data(j))/(max_data(j)-min_data(j));
end
end


%normalisation
for i=1:1:size(test_data,1)
for j=1:1:size(test_data,2)-1
    test_data(i,j)=(test_data(i,j)-min_data(j))/(max_data(j)-min_data(j));
end
end



no_of_data_items_in_each_class=zeros(no_of_classes,1);
my_class(no_of_classes).data=zeros(1,size(training_data,2));
for i=1:1:floor(fraction*size_x)
class_no=training_data(i,class_label);
no_of_data_items_in_each_class(class_no,1)=no_of_data_items_in_each_class(class_no,1)+1;
my_class(class_no).data(no_of_data_items_in_each_class(class_no),:)=training_data(i,:);
end

class_vector=zeros(floor(fraction*size_x),1);
count=1;
for class_no=1:1:no_of_classes
    for i=1:1:no_of_data_items_in_each_class(class_no)
    class_vector(count,1)=my_class(class_no).data(i,size_y);
    count=count+1;
    end
end


s=zeros(no_of_classes+1,1);
for i=1:1:no_of_classes
s(i+1,1)=s(i,1)+no_of_data_items_in_each_class(i);
end



for i=1:1:no_of_classes
temp_cluster_vector=my_multivariate_k_means1(my_class(i).data,floor(no_of_clusters/no_of_classes));
temp_cluster_vector=temp_cluster_vector+floor(no_of_clusters/no_of_classes)*(i-1);
[cluster_vector(s(i)+1:s(i+1),1)]=temp_cluster_vector;
end


        %%%%%%%%%clustered training_data%%%%%%%%%%%%%%%%%%%%%%
clear training_data;
for i=1:1:no_of_classes
    training_data(s(i)+1:s(i+1),:)=my_class(i).data;
end
training_data(:,size_y)=cluster_vector;
rnd_vector=randperm(length(training_data));
training_data=training_data(rnd_vector,:);
class_vector=class_vector(rnd_vector,:);
cluster_vector=cluster_vector(rnd_vector,:);

        %%%%%%%%%%%%class_cluster_cooccurance_matrix%%%%%%%%%%%%%%%
class_cluster_cooccurance_matrix=zeros(no_of_clusters,no_of_classes);
for i=1:1:length(cluster_vector)
    class_cluster_cooccurance_matrix(cluster_vector(i),class_vector(i))=class_cluster_cooccurance_matrix(cluster_vector(i),class_vector(i))+1;
end
class_cluster_cooccurance_matrix
sum(class_cluster_cooccurance_matrix)

        %%%%%%%%%%%%target_training_cluster_matrix%%%%%%%%%%%%%%%%
target_training_cluster_matrix=zeros(length(training_data),no_of_clusters);
for i=1:1:length(training_data)
target_training_cluster_matrix(i,training_data(i,size_y))=1;
end

        %%%%%%%%%%%%%%%%%target_class_matrix%%%%%%%%%%%%%%%%%
target_training_class_matrix=zeros(length(training_data),no_of_classes);
for i=1:1:length(training_data)
target_training_class_matrix(i,class_vector(i,1))=1;
end
target_testing_class_matrix=zeros(length(test_data),no_of_classes);
class_vector1=test_data(:,size(test_data,2));
for i=1:1:length(test_data)
target_testing_class_matrix(i,class_vector1(i,1))=1;
end
accuracy=zeros(15,1);


