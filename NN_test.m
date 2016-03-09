
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


%obtained_cluster_confidence_vector=zeros(length(obtained_cluster_confidence_vector_temp),no_of_clusters);
test_cluster_obtained=zeros(size(neural_network_testing_input,1),1);
for i=1:1:length(obtained_cluster_confidence_vector_temp)
    cl=find(obtained_cluster_confidence_vector_temp(i,:)==max(obtained_cluster_confidence_vector_temp(i,:)));
    %obtained_cluster_confidence_vector(i,cl(1,1))=1;
    test_cluster_obtained(i,1)=cl(1,1);
    
end
obtained_cluster_confidence_vector_temp
%test_cluster_obtained
