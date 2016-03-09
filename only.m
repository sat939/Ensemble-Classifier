clear
clc
acc=0;
for round=1:1:30
ordered_data=load('iris.txt');
[size_x size_y]=size(ordered_data);
data=ordered_data(randperm(size_x),:);
training_data=data(1:floor(0.9*size_x),:);
test_data=data(floor(0.9*size_x)+1:size_x,:);
class_vector=training_data(:,size_y);
no_of_classes=length(unique(class_vector));

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

for k=1:1:15
clear group;
group=training_data(:,size(test_data,2));
class_confidence_matrix_temp1 = my_knnclassify_k(training_data(:,1:size(training_data,2)-1),training_data(:,1:size(training_data,2)-1),group,k,no_of_classes);
clear sum;
train_class_obtained1=zeros(length(class_confidence_matrix_temp1),1);
for row=1:1:length(training_data)
train_class_obtained1(row,1)=find(class_confidence_matrix_temp1(row,:)==max(class_confidence_matrix_temp1(row,:)),1);
end
class_confidence_matrix_temp1
correct=0;
chk=[class_vector train_class_obtained1];
for i=1:1:length(chk)
    if chk(i,1)==chk(i,2)
        correct=correct+1;
    end
end
%class_confidence_matrix_temp1
accuracy(k,1)=correct*100/length(chk);
end
accuracy
sum(accuracy)/15
k=find(accuracy==max(accuracy),1)
accuracy(k,1)
class_confidence_matrix_temp1 = my_knnclassify_k(training_data(:,1:size(training_data,2)-1),training_data(:,1:size(training_data,2)-1),group,k,no_of_classes);





clear group;
group=training_data(:,size(training_data,2));
obtained_class_confidence_vector1_temp = my_knnclassify_k(test_data(:,1:size(test_data,2)-1),training_data(:,1:size(training_data,2)-1),group,k,no_of_classes);
clear sum
test_class_obtained1=zeros(length(obtained_class_confidence_vector1_temp),1);
for row=1:1:length(test_data)
    test_class_obtained1(row,1)=find(obtained_class_confidence_vector1_temp(row,:)==max(obtained_class_confidence_vector1_temp(row,:)),1);
end

obtained_class_confidence_vector1_temp

clear chk
chk=[test_class_obtained1 test_data(:,size_y)]
correct=0;
for i=1:1:size(chk,1)
if chk(i,1)==chk(i,2)
    correct=correct+1;
end
end
op=correct*100/(size(chk,1))
acc(size(acc)+1,1)=op;
end
acc

y=(acc(2:31));
sum(y)/30