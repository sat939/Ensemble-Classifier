clear
clc
accuracies_for_various_runs=0;
for round=1:1:8
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
accuracy(k,1)=correct*100/length(chk);
end
accuracy
sum(accuracy)/15
k=find(accuracy==max(accuracy),1)
accuracy(k,1)
cluster_confidence_matrix_temp1 = my_knnclassify_k(training_data(:,1:size(training_data,2)-1),training_data(:,1:size(training_data,2)-1),group,k,no_of_clusters);






clear group;
group=training_data(:,size(training_data,2));
obtained_cluster_confidence_vector1_temp = my_knnclassify_k(test_data(:,1:size(test_data,2)-1),training_data(:,1:size(training_data,2)-1),group,k,no_of_clusters);
clear sum
test_cluster_obtained1=zeros(length(obtained_cluster_confidence_vector1_temp),1);
for row=1:1:length(test_data)
    test_cluster_obtained1(row,1)=find(obtained_cluster_confidence_vector1_temp(row,:)==max(obtained_cluster_confidence_vector1_temp(row,:)),1);
end

obtained_cluster_confidence_vector1_temp






neural_network_training_input=training_data(:,1:size(training_data,2)-1);
neural_network_training_output=target_training_cluster_matrix;
learning_rate=0.9%input('enter learning rate of neural network classifeir.. \n 0.9 is preferable\n');
no_of_hidden_layers=1%input('enter no of hidden layers\n');
no_of_units_in_each_layer=zeros(1,no_of_hidden_layers+2);
no_of_units_in_each_layer(1,1)=size(training_data,2)-1;
no_of_units_in_each_layer(1,no_of_hidden_layers+2)=no_of_clusters;
for i=1:1:no_of_hidden_layers
    if i==1
no_of_units_in_each_layer(1,1+i)=5%input('enter no of units in 1st hidden layer \n');
    else
no_of_units_in_each_layer(1,1+i)=input('enter no of units in the next hidden layer \n');
    end
end

tuples_misclassified_classified_treshold=9%input('enter the desired accuracy of the training data classification so as to set the terminating condition\n 85 and above is preferable\n');
desired_epoch=100%input('enter the maximum epoch so as to set the terminating condition\n an epoch of around 125 is fine\n');
desired_max_delta_weight=0.00008%input('enter the desired_max_delta_weight so as to set the terminating condition\n some thing around 0.0008 must be fine\n');


 sample_input_layer_node.weight(1,:)=zeros(1,no_of_units_in_each_layer(:,2));
 sample_input_layer_node.error=0;
 sample_input_layer_node.bias=0;
 sample_input_layer_node.input=0;
 sample_input_layer_node.output=0;
 sample_input_layer_node.delta_weight(1,:)=zeros(1,no_of_units_in_each_layer(:,2));

 i=no_of_hidden_layers;
 sample_hidden_layer_node(i).weight(1,:)=zeros(1,no_of_units_in_each_layer(:,i+2));
 sample_hidden_layer_node(i).error=0;
 sample_hidden_layer_node(i).bias=0;
 sample_hidden_layer_node(i).input=0;
 sample_hidden_layer_node(i).output=0;
 sample_hidden_layer_node(i).delta_weight(1,:)=zeros(1,no_of_units_in_each_layer(:,i+2));
 
 for i=1:1:no_of_hidden_layers
 sample_hidden_layer_node(i).weight(1,:)=zeros(1,no_of_units_in_each_layer(:,i+2));
 sample_hidden_layer_node(i).error=0;
 sample_hidden_layer_node(i).bias=0;
 sample_hidden_layer_node(i).input=0;
 sample_hidden_layer_node(i).output=0;
 sample_hidden_layer_node(i).delta_weight(1,:)=zeros(1,no_of_units_in_each_layer(:,i+2));
 end

  

 sample_output_layer_node.error=0;
 sample_output_layer_node.bias=0;
 sample_output_layer_node.input=0;
 sample_output_layer_node.output=0;


 
input_layer_unit(no_of_units_in_each_layer(:,1))=sample_input_layer_node;
for i=1:1:no_of_units_in_each_layer(:,1)
input_layer_unit(i)=sample_input_layer_node;
end

hidden_layer(no_of_hidden_layers).unit(1)=sample_hidden_layer_node(1);
for hlno=1:1:no_of_hidden_layers
for i=1:1:no_of_units_in_each_layer(:,hlno+1)
hidden_layer(hlno).unit(i)=sample_hidden_layer_node(hlno);
end
end

output_layer_unit(no_of_units_in_each_layer(:,no_of_hidden_layers+2))=sample_output_layer_node;
for i=1:1:no_of_units_in_each_layer(:,no_of_hidden_layers+2)
output_layer_unit(i)=sample_output_layer_node;
end

for i=1:1:no_of_units_in_each_layer(:,1)
input_layer_unit(i).delta_weight=zeros(no_of_units_in_each_layer(:,2),1);
input_layer_unit(i).weight=(-0.5)+rand(no_of_units_in_each_layer(:,2),1);
input_layer_unit(i).bias=(-0.5)+rand;
end


for hlno=1:1:no_of_hidden_layers
for i=1:1:no_of_units_in_each_layer(:,1+hlno)
hidden_layer(hlno).unit(i).delta_weight=zeros(no_of_units_in_each_layer(:,hlno+2),1);
hidden_layer(hlno).unit(i).weight=(-0.5)+rand(no_of_units_in_each_layer(:,hlno+2),1);
hidden_layer(hlno).unit(i).bias=(-0.5)+rand;
end
end

for i=1:1:no_of_units_in_each_layer(:,no_of_hidden_layers+2)
output_layer_unit(i).bias=(-0.5)+rand;
end



epoch=0;
terminating_condition=0;
cluster_confidence_matrix_temp=zeros(length(training_data),no_of_clusters);
cluster_confidence_matrix=zeros(length(training_data),no_of_clusters);
while ~terminating_condition 
epoch=epoch+1;
fprintf('the current epoch is %d',epoch);
     
for row=1:1:length(training_data)
    
   %input/output  

for i=1:1:no_of_units_in_each_layer(:,1)

input_layer_unit(i).input=neural_network_training_input(row,i);
input_layer_unit(i).output=input_layer_unit(i).input;

end

for hlno=1:1:no_of_hidden_layers
for i=1:1:no_of_units_in_each_layer(:,hlno+1)
sum=0;
    if hlno==1
for j=1:1:no_of_units_in_each_layer(:,hlno)
sum=sum+input_layer_unit(j).output*input_layer_unit(j).weight(i);
end
    else
for j=1:1:no_of_units_in_each_layer(:,hlno)
sum=sum+hidden_layer(hlno-1).unit(j).output*hidden_layer(hlno-1).unit(j).weight(i);
end
    end
    
hidden_layer(hlno).unit(i).input=sum+hidden_layer(hlno).unit(i).bias;
hidden_layer(hlno).unit(i).output=1/(1+exp(-1*hidden_layer(hlno).unit(i).input));

end
end

for i=1:1:no_of_units_in_each_layer(:,no_of_hidden_layers+2)

sum=0;
for j=1:1:no_of_units_in_each_layer(:,no_of_hidden_layers+1)
sum=sum+hidden_layer(no_of_hidden_layers).unit(j).output*hidden_layer(no_of_hidden_layers).unit(j).weight(i);
end
output_layer_unit(i).input=sum+output_layer_unit(i).bias;
output_layer_unit(i).output=1/(1+exp(-1*output_layer_unit(i).input));

end



for i=1:1:no_of_units_in_each_layer(:,no_of_hidden_layers+2)
cluster_confidence_matrix_temp(row,i)=output_layer_unit(i).output;
end


        %%%%%%error%%%%%%%%%
for i=1:1:no_of_units_in_each_layer(:,no_of_hidden_layers+2)
output_layer_unit(i).error=(output_layer_unit(i).output)*(1-output_layer_unit(i).output)*(neural_network_training_output(row,i)-output_layer_unit(i).output);
end

for hlno=no_of_hidden_layers:-1:1
for i=1:1:no_of_units_in_each_layer(:,hlno+1)
sum=0;
for j=1:1:no_of_units_in_each_layer(:,hlno+2)
if hlno==no_of_hidden_layers
    sum=sum+output_layer_unit(j).error*hidden_layer(hlno).unit(i).weight(j);
else
    sum=sum+hidden_layer(hlno+1).unit(j).error*hidden_layer(hlno).unit(i).weight(j);
end
end
hidden_layer(hlno).unit(i).error=(hidden_layer(hlno).unit(i).output)*(1-(hidden_layer(hlno).unit(i).output))*sum;
end
end

for i=1:1:no_of_units_in_each_layer(1,1)
    x=0;
    for j=1:1:no_of_units_in_each_layer(1,2)
    x=x+hidden_layer(1).unit(j).error*input_layer_unit(i).weight(j);
    end
    input_layer_unit(i).error=input_layer_unit(i).output*(1-input_layer_unit(i).output)*x;
end





    %weights
for i=1:1:no_of_units_in_each_layer(:,1)
for j=1:1:no_of_units_in_each_layer(:,2)

input_layer_unit(i).delta_weight(j)=learning_rate*hidden_layer(1).unit(j).error*input_layer_unit(i).output;
input_layer_unit(i).weight(j)=input_layer_unit(i).weight(j)+input_layer_unit(i).delta_weight(j);


end
end

for hlno=1:1:no_of_hidden_layers
for i=1:1:no_of_units_in_each_layer(:,hlno+1)
for j=1:1:no_of_units_in_each_layer(:,hlno+2)
    
if hlno<no_of_hidden_layers    
hidden_layer(hlno).unit(i).delta_weight(j)=learning_rate*hidden_layer(hlno+1).unit(j).error*hidden_layer(hlno).unit(i).output;
hidden_layer(hlno).unit(i).weight(j)=hidden_layer(hlno).unit(i).weight(j)+hidden_layer(hlno).unit(i).delta_weight(j);
else    
hidden_layer(hlno).unit(i).delta_weight(j)=learning_rate*output_layer_unit(j).error*hidden_layer(hlno).unit(i).output;
hidden_layer(hlno).unit(i).weight(j)=hidden_layer(hlno).unit(i).weight(j)+hidden_layer(hlno).unit(i).delta_weight(j);
end

end
end
end





    %biases
for i=1:1:no_of_units_in_each_layer(:,1)
delta_bias=learning_rate*input_layer_unit(i).error;
input_layer_unit(i).bias=input_layer_unit(i).bias+delta_bias;
end

for hlno=1:1:no_of_hidden_layers
for i=1:1:no_of_units_in_each_layer(:,hlno+1)
delta_bias=learning_rate*hidden_layer(hlno).unit(i).error;
hidden_layer(hlno).unit(i).bias=hidden_layer(hlno).unit(i).bias+delta_bias;
end
end

for i=1:1:no_of_units_in_each_layer(:,no_of_hidden_layers+2)
delta_bias=learning_rate*output_layer_unit(i).error;	
output_layer_unit(i).bias=output_layer_unit(i).bias+delta_bias;
end

 
end


clear cluster_confidence_matrix;
train_cluster_obtained=zeros(length(cluster_confidence_matrix_temp),1);
for i=1:1:length(cluster_confidence_matrix_temp)
    cl=find(cluster_confidence_matrix_temp(i,:)==max(cluster_confidence_matrix_temp(i,:)),1);
    %cluster_confidence_matrix(i,cl(1,1))=1;
    train_cluster_obtained(i,1)=cl;
    
end

correct=0;
incorrect=0;
    %clusterifier accuracy
    chk=[training_data(:,size(training_data,2)) train_cluster_obtained];
for i=1:1:length(training_data)
    if chk(i,1)==chk(i,2)
        correct=correct+1;
    else 
        incorrect=incorrect+1;
    end
    
end
tuples_misclassified_classified=100*incorrect/length(training_data)



    %delta_weight terminating condition 
    clear sum;
    dw=zeros(max(no_of_units_in_each_layer(1,1:2+no_of_hidden_layers)),sum(no_of_units_in_each_layer(1,1:1+no_of_hidden_layers)));
    dw_col=1;
    for dw_col=1:1:no_of_units_in_each_layer(1,1)
        dw(1:length(input_layer_unit(dw_col).delta_weight),dw_col)=input_layer_unit(dw_col).delta_weight;
    end
    
    
    for hlno=1:1:no_of_hidden_layers
    for i=1:1:no_of_units_in_each_layer(1,1+hlno)
        dw_col=dw_col+1;
        dw(1:length(hidden_layer(hlno).unit(i).delta_weight),dw_col)=hidden_layer(hlno).unit(i).delta_weight;
    end
    end
    
    dw_max=max(max(abs(dw)));
    
    
    
    % checking for termination condition
    clear cond;
    cond=zeros(1,3);
    if tuples_misclassified_classified<=tuples_misclassified_classified_treshold
        cond(1,1)=1;
    end

    if dw_max<desired_max_delta_weight
        cond(1,2)=1;
    end
    
    if epoch>=desired_epoch
        cond(1,3)=1;
    end
    %tuples_misclassified_classified
    
    
    if ~isempty(find(cond==1, 1))
        terminating_condition=1;
    else 
        terminating_condition=0;
    end
    
    
end


disp('end of training');
cluster_confidence_matrix_temp









neural_network_testing_input=test_data(:,1:size(test_data,2)-1);
obtained_cluster_confidence_vector_temp=zeros(length(test_data),no_of_clusters);
                     



for test_it=1:1:length(test_data)

for i=1:1:no_of_units_in_each_layer(:,1)

input_layer_unit(i).input=neural_network_testing_input(test_it,i);
input_layer_unit(i).output=input_layer_unit(i).input;

end

for hlno=1:1:no_of_hidden_layers
for i=1:1:no_of_units_in_each_layer(:,hlno+1)
sum=0;
    if hlno==1
for j=1:1:no_of_units_in_each_layer(:,hlno)
sum=sum+input_layer_unit(j).output*input_layer_unit(j).weight(i);
end
    else
for j=1:1:no_of_units_in_each_layer(:,hlno)
sum=sum+hidden_layer(hlno-1).unit(j).output*hidden_layer(hlno-1).unit(j).weight(i);
end
    end
    
hidden_layer(hlno).unit(i).input=sum+hidden_layer(hlno).unit(i).bias;
hidden_layer(hlno).unit(i).output=1/(1+exp(-1*hidden_layer(hlno).unit(i).input));

end
end

for i=1:1:no_of_units_in_each_layer(:,no_of_hidden_layers+2)

sum=0;
for j=1:1:no_of_units_in_each_layer(:,no_of_hidden_layers+1)
sum=sum+hidden_layer(no_of_hidden_layers).unit(j).output*hidden_layer(no_of_hidden_layers).unit(j).weight(i);
end
output_layer_unit(i).input=sum+output_layer_unit(i).bias;
output_layer_unit(i).output=1/(1+exp(-1*output_layer_unit(i).input));

end


for i=1:1:no_of_units_in_each_layer(:,no_of_hidden_layers+2)
obtained_cluster_confidence_vector_temp(test_it,i)=output_layer_unit(i).output;
end

end


test_cluster_obtained=zeros(size(neural_network_testing_input,1),1);
for i=1:1:length(obtained_cluster_confidence_vector_temp)
    cl=find(obtained_cluster_confidence_vector_temp(i,:)==max(obtained_cluster_confidence_vector_temp(i,:)));
    test_cluster_obtained(i,1)=cl(1,1);    
end
obtained_cluster_confidence_vector_temp





neural_network_fusion_training_input=zeros(size(cluster_confidence_matrix_temp,1),2*size(cluster_confidence_matrix_temp,2));
neural_network_fusion_training_input(:,1:size(cluster_confidence_matrix_temp,2))=cluster_confidence_matrix_temp;
neural_network_fusion_training_input(:,size(cluster_confidence_matrix_temp,2)+1:2*size(cluster_confidence_matrix_temp,2))=cluster_confidence_matrix_temp1;
neural_network_fusion_training_output=target_training_class_matrix;

f_learning_rate=0.9%input('enter learning rate of neural network classifeir.. \n 0.9 is preferable\n');
f_no_of_hidden_layers=1%input('enter no of hidden layers\n');
f_no_of_units_in_each_layer=zeros(1,f_no_of_hidden_layers+2);
f_no_of_units_in_each_layer(1,1)=size(neural_network_fusion_training_input,2);
f_no_of_units_in_each_layer(1,f_no_of_hidden_layers+2)=no_of_classes;
for i=1:1:f_no_of_hidden_layers
    if i==1
f_no_of_units_in_each_layer(1,1+i)=5%input('enter no of units in 1st hidden layer \n');
    else
f_no_of_units_in_each_layer(1,1+i)=input('enter no of units in the next hidden layer \n');
    end
end

f_tuples_misclassified_classified_treshold=1%input('enter the desired accuracy of the training data classification so as to set the terminating condition\n 85 and above is preferable\n');
f_desired_epoch=15%input('enter the maximum epoch so as to set the terminating condition\n an epoch of around 125 is fine\n');
f_desired_max_delta_weight=0.0008%input('enter the f_desired_max_delta_weight so as to set the terminating condition\n some thing around 0.0008 must be fine\n');




        %%%%%%%%%normalisation%%%%%%%%%%%%
for j=1:1:no_of_clusters
min_data(j)=min(neural_network_fusion_training_input(:,j));
max_data(j)=max(neural_network_fusion_training_input(:,j));
end;

for i=1:1:length(neural_network_fusion_training_input);
for j=1:1:no_of_clusters
   neural_network_fusion_training_input(i,j)=(neural_network_fusion_training_input(i,j)-min_data(j))/(max_data(j)-min_data(j));
end
end




 f_sample_input_layer_node.weight(1,:)=zeros(1,f_no_of_units_in_each_layer(:,2));
 f_sample_input_layer_node.error=0;
 f_sample_input_layer_node.bias=0;
 f_sample_input_layer_node.input=0;
 f_sample_input_layer_node.output=0;
 f_sample_input_layer_node.delta_weight(1,:)=zeros(1,f_no_of_units_in_each_layer(:,2));

 for i=1:1:f_no_of_hidden_layers
 f_sample_hidden_layer_node(i).weight(1,:)=zeros(1,f_no_of_units_in_each_layer(:,i+2));
 f_sample_hidden_layer_node(i).error=0;
 f_sample_hidden_layer_node(i).bias=0;
 f_sample_hidden_layer_node(i).input=0;
 f_sample_hidden_layer_node(i).output=0;
 f_sample_hidden_layer_node(i).delta_weight(1,:)=zeros(1,f_no_of_units_in_each_layer(:,i+2));
 
end

  
 f_sample_output_layer_node.error=0;
 f_sample_output_layer_node.bias=0;
 f_sample_output_layer_node.input=0;
 f_sample_output_layer_node.output=0;


for i=1:1:f_no_of_units_in_each_layer(:,1)
f_input_layer_unit(i)=f_sample_input_layer_node;
end

for hlno=1:1:f_no_of_hidden_layers
for i=1:1:f_no_of_units_in_each_layer(:,hlno+1)
f_hidden_layer(hlno).unit(i)=f_sample_hidden_layer_node(hlno);
end
end


for i=1:1:f_no_of_units_in_each_layer(:,f_no_of_hidden_layers+2)
f_output_layer_unit(i)=f_sample_output_layer_node;
end

for i=1:1:f_no_of_units_in_each_layer(:,1)

f_input_layer_unit(i).weight=(-0.5)+rand(f_no_of_units_in_each_layer(:,2),1);
f_input_layer_unit(i).bias=(-0.5)+rand;

end


for hlno=1:1:f_no_of_hidden_layers
for i=1:1:f_no_of_units_in_each_layer(:,1+hlno)

f_hidden_layer(hlno).unit(i).weight=(-0.5)+rand(f_no_of_units_in_each_layer(:,hlno+2),1);
f_hidden_layer(hlno).unit(i).bias=(-0.5)+rand;

end
end

for i=1:1:f_no_of_units_in_each_layer(:,f_no_of_hidden_layers+2)

f_output_layer_unit(i).bias=(-0.5)+rand;

end



epoch=0;
terminating_condition=0;
while ~terminating_condition 
epoch=epoch+1;
fprintf('the current epoch is %d',epoch);
     
for row=1:1:length(neural_network_fusion_training_input)

   

    %input/output
for i=1:1:f_no_of_units_in_each_layer(:,1)

f_input_layer_unit(i).input=neural_network_fusion_training_input(row,i);
f_input_layer_unit(i).output=f_input_layer_unit(i).input;

end

for hlno=1:1:f_no_of_hidden_layers
for i=1:1:f_no_of_units_in_each_layer(:,1+hlno)
sum=0;
if hlno==1
    for j=1:1:f_no_of_units_in_each_layer(:,hlno)
    sum=sum+f_input_layer_unit(j).output*f_input_layer_unit(j).weight(i);
    end
else
    for j=1:1:f_no_of_units_in_each_layer(:,hlno)
    sum=sum+f_hidden_layer(hlno-1).unit(j).output*f_hidden_layer(hlno-1).unit(j).weight(i);
    end   
end
f_hidden_layer(hlno).unit(i).input=sum+f_hidden_layer(hlno).unit(i).bias;
f_hidden_layer(hlno).unit(i).output=(1)/(1+exp(-1*f_hidden_layer(hlno).unit(i).input));
end
end

for i=1:1:f_no_of_units_in_each_layer(:,f_no_of_hidden_layers+2)

sum=0;
for j=1:1:f_no_of_units_in_each_layer(:,f_no_of_hidden_layers+1)
sum=sum+f_hidden_layer(f_no_of_hidden_layers).unit(j).output*f_hidden_layer(f_no_of_hidden_layers).unit(j).weight(i);
end
f_output_layer_unit(i).input=sum+f_output_layer_unit(i).bias;
f_output_layer_unit(i).output=1/(1+exp(-1*f_output_layer_unit(i).input));

end



for i=1:1:f_no_of_units_in_each_layer(:,f_no_of_hidden_layers+2)
class_confidence_matrix_temp(row,i)=f_output_layer_unit(i).output;
end



    %error
for i=1:1:f_no_of_units_in_each_layer(:,f_no_of_hidden_layers+2)
f_output_layer_unit(i).error=(f_output_layer_unit(i).output)*(1-f_output_layer_unit(i).output)*(neural_network_fusion_training_output(row,i)-f_output_layer_unit(i).output);
end

for hlno=f_no_of_hidden_layers:-1:1
for i=1:1:f_no_of_units_in_each_layer(:,hlno+1)
sum=0;
for j=1:1:f_no_of_units_in_each_layer(:,hlno+2)
if hlno==f_no_of_hidden_layers
    sum=sum+f_output_layer_unit(j).error*f_hidden_layer(hlno).unit(i).weight(j);
else
    sum=sum+f_hidden_layer(hlno+1).unit(j).error*f_hidden_layer(hlno).unit(i).weight(j);
end
end
f_hidden_layer(hlno).unit(i).error=(f_hidden_layer(hlno).unit(i).output)*(1-(f_hidden_layer(hlno).unit(i).output))*sum;
end
end

for i=1:1:f_no_of_units_in_each_layer(1,1)
    x=0;
    for j=1:1:f_no_of_units_in_each_layer(1,2)
    x=x+f_hidden_layer(1).unit(j).error*f_input_layer_unit(i).weight(j);
    end
    f_input_layer_unit(i).error=f_input_layer_unit(i).output*(1-f_input_layer_unit(i).output)*x;
end





    %weight
for i=1:1:f_no_of_units_in_each_layer(:,1)
for j=1:1:f_no_of_units_in_each_layer(:,2)

delta_weight=f_learning_rate*f_hidden_layer(1).unit(j).error*f_input_layer_unit(i).output;
f_input_layer_unit(i).weight(j)=f_input_layer_unit(i).weight(j)+delta_weight;


end
end

for hlno=1:1:f_no_of_hidden_layers
for i=1:1:f_no_of_units_in_each_layer(:,hlno+1)
for j=1:1:f_no_of_units_in_each_layer(:,hlno+2)

if hlno<f_no_of_hidden_layers    
delta_weight=f_learning_rate*f_hidden_layer(hlno+1).unit(j).error*f_hidden_layer(hlno).unit(i).output;
f_hidden_layer(hlno).unit(i).weight(j)=f_hidden_layer(hlno).unit(i).weight(j)+delta_weight;
else    
delta_weight=f_learning_rate*f_output_layer_unit(j).error*f_hidden_layer(hlno).unit(i).output;
f_hidden_layer(hlno).unit(i).weight(j)=f_hidden_layer(hlno).unit(i).weight(j)+delta_weight;
end

end
end
end




    %bias
for i=1:1:f_no_of_units_in_each_layer(:,1)
delta_bias=f_learning_rate*f_input_layer_unit(i).error;
f_input_layer_unit(i).bias=f_input_layer_unit(i).bias+delta_bias;
end

for hlno=1:1:f_no_of_hidden_layers
for i=1:1:f_no_of_units_in_each_layer(:,hlno+1)
delta_bias=f_learning_rate*f_hidden_layer(hlno).unit(i).error;
f_hidden_layer(hlno).unit(i).bias=f_hidden_layer(hlno).unit(i).bias+delta_bias;
end
end

for i=1:1:f_no_of_units_in_each_layer(:,f_no_of_hidden_layers+2)
delta_bias=f_learning_rate*f_output_layer_unit(i).error;	
f_output_layer_unit(i).bias=f_output_layer_unit(i).bias+delta_bias;
end


end


clear class_confidence_matrix;
class_obtained=zeros(length(class_confidence_matrix_temp),1);
for i=1:1:length(class_confidence_matrix_temp)
    cl=find(class_confidence_matrix_temp(i,:)==max(class_confidence_matrix_temp(i,:)),1);
    %cluster_confidence_matrix(i,cl(1,1))=1;
    class_obtained(i,1)=cl;    
end

incorrect=0;
correct=0;
    %classifier accuracy
    chk=[class_vector class_obtained];
for i=1:1:length(training_data)
    if chk(i,1)==chk(i,2)
        correct=correct+1;
    else 
        incorrect=incorrect+1;
    end
end
f_tuples_misclassified_classified=100*incorrect/length(training_data)



    %delta_weight terminating condition 
    clear sum;
    dw=zeros(max(f_no_of_units_in_each_layer(1,1:2+f_no_of_hidden_layers)),sum(f_no_of_units_in_each_layer(1,1:1+f_no_of_hidden_layers)));
    dw_col=1;
    for dw_col=1:1:f_no_of_units_in_each_layer(1,1)
        dw(1:length(f_input_layer_unit(dw_col).delta_weight),dw_col)=f_input_layer_unit(dw_col).delta_weight;
    end
    
    
    for hlno=1:1:f_no_of_hidden_layers
    for i=1:1:f_no_of_units_in_each_layer(1,1+hlno)
        dw_col=dw_col+1;
        dw(1:length(f_hidden_layer(hlno).unit(i).delta_weight),dw_col)=f_hidden_layer(hlno).unit(i).delta_weight;
    end
    end
    
    dw_max=max(max(abs(dw)));
    
    
    
    % checking for termination condition
    clear cond;
    cond=zeros(1,3);
    if f_tuples_misclassified_classified<=f_tuples_misclassified_classified_treshold
        cond(1,1)=1;
    end

    %if dw_max<f_desired_max_delta_weight
    %    cond(1,2)=1;
    %end
    cond(1,2)=0;
    if epoch>=f_desired_epoch
        cond(1,3)=1;
    end
    
    
    if ~isempty(find(cond==1, 1))
        terminating_condition=1;
    else 
        terminating_condition=0;
    end
    


end

disp('end of training');
class_confidence_matrix_temp











neural_network_fusion_testing_input=zeros(size(obtained_cluster_confidence_vector_temp,1),2*size(obtained_cluster_confidence_vector_temp,2));
neural_network_fusion_testing_input(:,1:size(obtained_cluster_confidence_vector_temp,2))=obtained_cluster_confidence_vector_temp;
neural_network_fusion_testing_input(:,size(obtained_cluster_confidence_vector_temp,2)+1:2*size(obtained_cluster_confidence_vector_temp,2))=obtained_cluster_confidence_vector1_temp;
for row=1:1:length(neural_network_fusion_testing_input)

     %input/output
for i=1:1:f_no_of_units_in_each_layer(:,1)

f_input_layer_unit(i).input=neural_network_fusion_testing_input(row,i);
f_input_layer_unit(i).output=f_input_layer_unit(i).input;

end

for hlno=1:1:f_no_of_hidden_layers
for i=1:1:f_no_of_units_in_each_layer(:,1+hlno)
sum=0;
if hlno==1
    for j=1:1:f_no_of_units_in_each_layer(:,hlno)
    sum=sum+f_input_layer_unit(j).output*f_input_layer_unit(j).weight(i);
    end
else
    for j=1:1:f_no_of_units_in_each_layer(:,hlno)
    sum=sum+f_hidden_layer(hlno-1).unit(j).output*f_hidden_layer(hlno-1).unit(j).weight(i);
    end   
end
f_hidden_layer(hlno).unit(i).input=sum+f_hidden_layer(hlno).unit(i).bias;
f_hidden_layer(hlno).unit(i).output=(1)/(1+exp(-1*f_hidden_layer(hlno).unit(i).input));
end
end

for i=1:1:f_no_of_units_in_each_layer(:,f_no_of_hidden_layers+2)

sum=0;
for j=1:1:f_no_of_units_in_each_layer(:,f_no_of_hidden_layers+1)
sum=sum+f_hidden_layer(f_no_of_hidden_layers).unit(j).output*f_hidden_layer(f_no_of_hidden_layers).unit(j).weight(i);
end
f_output_layer_unit(i).input=sum+f_output_layer_unit(i).bias;
f_output_layer_unit(i).output=1/(1+exp(-1*f_output_layer_unit(i).input));

end


for i=1:1:no_of_classes
obtained_class_confidence_vector_temp(row,i)=f_output_layer_unit(i).output;
end

end
obtained_class_confidence_vector_temp
obtained_class_confidence_vector=zeros(size(test_data,1),no_of_classes);
for occv=1:1:size(test_data,1)
    test_class_obtained(occv,1)=find(obtained_class_confidence_vector_temp(occv,:)==max(obtained_class_confidence_vector_temp(occv,:)));
end



[test_class_obtained class_vector1]

test_class_correct=0;
test_class_incorrect=0;

tctc=test_data(:,size(test_data,2));
for i=1:1:size(test_data,1)
    if test_class_obtained(i,1)==tctc(i,1);
        test_class_correct=test_class_correct+1;
    else
        test_class_incorrect=test_class_incorrect+1;
    end
    
end

test_class_accuracy=test_class_correct*100/(test_class_correct+test_class_incorrect);
test_class_accuracy
accuracies_for_various_runs(length(accuracies_for_various_runs)+1,1)=test_class_accuracy;
disp('end of testing');




end


clear sum
y=accuracies_for_various_runs(2:9);
sum(y)/9










