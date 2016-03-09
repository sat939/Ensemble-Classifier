function [ new_cluster_vector ] = my_multivariate_k_means1( training_data, no_of_clusters )

[size_x size_y]=size(training_data);



        %%%%%%%%%%initialize clusters%%%%%%%%%%%%% 
new_cluster_vector=zeros(size_x,1);
my_cluster(no_of_clusters).mean=0;
my_cluster(no_of_clusters).indices=0;
my_cluster(no_of_clusters).count=0;
        for i=1:1:no_of_clusters
        my_cluster(i).indices=ceil(rand*size_x);
        new_cluster_vector(my_cluster(i).indices,1)=i;
        my_cluster(i).mean=training_data(my_cluster(i).indices,1:size(training_data,2));
        end
old_cluster_vector=new_cluster_vector;



        %%%%%%%assign each data item to nearest cluster%%%%%%%%%%%
epoch=0;        
while(1==1)
epoch=epoch+1;   
my_break=1;
    for row=1:1:size_x    
            if my_break==0
            break;
            end
            distance_to_each_cluster=zeros(1,no_of_clusters);
            for i=1:1:no_of_clusters

                    d=0;
                    for k=1:1:size_y-1
                    d=d+(training_data(row,k)-my_cluster(i).mean(1,k))^2;
                    end
                    distance_to_each_cluster(1,i)=sqrt(d);

            end  
            new_cluster_vector(row,1)=find(distance_to_each_cluster==min(distance_to_each_cluster),1);

    end

            %%%%%%%%%%%%centers(means) update%%%%%%%%%%%
    for i=1:1:no_of_clusters
    my_cluster(i).indices=find(new_cluster_vector==i);
    my_cluster(i).count=length(find(new_cluster_vector==i));
        if my_cluster(i).count~=0
            if my_cluster(i).count==1
            my_cluster(i).mean=training_data(my_cluster(i).indices,1:size(training_data,2))/my_cluster(i).count;
            else
            my_cluster(i).mean=sum(training_data(my_cluster(i).indices,1:size(training_data,2)))/my_cluster(i).count;
            end
        else 
        my_cluster(i).mean=zeros(1,size(training_data,2)-1);
        end
    end
            %%%%%%%%%%%terminating condition%%%%%%%%%%%%%
    if (length(find((new_cluster_vector==old_cluster_vector)==1))==size_x)   
    break;
    end

old_cluster_vector=new_cluster_vector;
end

end