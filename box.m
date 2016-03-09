
x=training_data(:,1:4)';
net = feedforwardnet(7,'trainlm');
t=cluster_vector;
Te=zeros(135,9);
for row=1:1:135
Te(row,t(row))=1;
end
T=Te';
net.trainFcn = 'trainrp';
net = train(net,x,T);
y = net(x);
cluster_confidence_matrix_temp=y';


clear cluster_confidence_matrix;
cluster_obtained=zeros(length(cluster_confidence_matrix_temp),1);
for i=1:1:length(cluster_confidence_matrix_temp)
    cl=find(cluster_confidence_matrix_temp(i,:)==max(cluster_confidence_matrix_temp(i,:)),1);
    cluster_obtained(i,1)=cl;
    
end

correct=0;
    %clusterifier accuracy
    chk=[cluster_vector cluster_obtained];
for i=1:1:length(training_data)
    if chk(i,1)==chk(i,2)
        correct=correct+1;
    end
end
cluster_accuracy=100*correct/length(training_data)
