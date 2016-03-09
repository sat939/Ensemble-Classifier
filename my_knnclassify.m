function [ class_knn ] = my_knnclassify( test_set,training_set,group )

[size_x_train size_y_train]=size(training_set);
[size_x_test size_y_test]=size(test_set);
for i=1:1:size_x_test

clear d;
d=zeros(1,size_x_train);
for j=1:1:size_x_train
d(1,j)=norm(test_set(i,:)-training_set(j,:));
end
x=find(d==min(d));
class_knn(i,1)=group(x,1);

end

