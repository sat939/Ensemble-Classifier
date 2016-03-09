clear
clc
acc=0;
for round=1:1:30
    clearvars -except acc
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



neural_network_training_input=training_data(:,1:size(training_data,2)-1);
neural_network_training_output=target_training_class_matrix;
learning_rate=0.9%input('enter learning rate of neural network classifeir.. \n 0.9 is preferable\n');
no_of_hidden_layers=1%input('enter no of hidden layers\n');
no_of_units_in_each_layer=zeros(1,no_of_hidden_layers+2);
no_of_units_in_each_layer(1,1)=size(training_data,2)-1;
no_of_units_in_each_layer(1,no_of_hidden_layers+2)=no_of_classes;
for i=1:1:no_of_hidden_layers
    if i==1
no_of_units_in_each_layer(1,1+i)=5%input('enter no of units in 1st hidden layer \n');
    else
no_of_units_in_each_layer(1,1+i)=input('enter no of units in the next hidden layer \n');
    end
end

tuples_misclassified_classified_treshold=5%input('enter the desired accuracy of the training data classification so as to set the terminating condition\n 85 and above is preferable\n');
desired_epoch=50%input('enter the maximum epoch so as to set the terminating condition\n an epoch of around 125 is fine\n');
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
class_confidence_matrix_temp=zeros(length(training_data),no_of_classes);
class_confidence_matrix=zeros(length(training_data),no_of_classes);
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
class_confidence_matrix_temp(row,i)=output_layer_unit(i).output;
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


clear class_confidence_matrix;
train_class_obtained=zeros(length(class_confidence_matrix_temp),1);
for i=1:1:length(class_confidence_matrix_temp)
    cl=find(class_confidence_matrix_temp(i,:)==max(class_confidence_matrix_temp(i,:)),1);
    %class_confidence_matrix(i,cl(1,1))=1;
    train_class_obtained(i,1)=cl;
    
end

correct=0;
incorrect=0;
    %classifier accuracy
    chk=[training_data(:,size(training_data,2)) train_class_obtained];
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
class_confidence_matrix_temp



neural_network_testing_input=test_data(:,1:size(test_data,2)-1);
obtained_class_confidence_vector_temp=zeros(length(test_data),no_of_classes);
                     



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
obtained_class_confidence_vector_temp(test_it,i)=output_layer_unit(i).output;
end

end


%obtained_class_confidence_vector=zeros(length(obtained_class_confidence_vector_temp),no_of_classs);
test_class_obtained=zeros(size(neural_network_testing_input,1),1);
for i=1:1:length(obtained_class_confidence_vector_temp)
    cl=find(obtained_class_confidence_vector_temp(i,:)==max(obtained_class_confidence_vector_temp(i,:)));
    %obtained_class_confidence_vector(i,cl(1,1))=1;
    test_class_obtained(i,1)=cl(1,1);
    
end
obtained_class_confidence_vector_temp
clear chk
chk=[test_class_obtained test_data(:,size_y)]
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