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