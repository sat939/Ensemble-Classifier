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

correct=0;
incorrect=0;
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