function split_file(varargin)
p = inputParser;
addParamValue(p,'all_data','LC_wo_genetic_od_id_norm.txt',@ischar);
addParamValue(p,'out','RF_dec8_LC_wo_gen',@ischar);
% addParamValue(p,'csv','no',@ischar);
% addParamValue(p,'missing_value','?');
addParamValue(p,'test_pc','20');
% addParamValue(p,'randm','shuffle',@ischar);

new_array = varargin(:);

for i =1:size(new_array,1)
     for j = 1:size(p.Parameters,2)
         param=p.Parameters{j};
         array_entry=new_array{i};
         first_check=strcat('-',param);
         second_check=strcat('--',param);
         if strcmp(array_entry,first_check)
            new_array{i}=param;
         end
         if strcmp(array_entry,second_check)
            new_array{i}=param;
         end
     end
end

% new_array(strcmp('yes',new_array))={'shuffle'};
% new_array(strcmp('no',new_array))={'default'};

parse(p,new_array{:});

data_file = (p.Results.all_data);
disp(data_file);
out = (p.Results.out);
tpc = (p.Results.test_pc);
tpc = str2double(tpc);
rownumbers = size(data_file,1);

chunk = floor(rownumbers*tpc/100);
disp(chunk);
data_file = readtable(data_file);
test = data_file(((floor(100/tpc)-1))*chunk+1:(floor(100/tpc))*chunk,:);
train = [data_file(1:((floor(100/tpc))-1)*chunk,:); data_file((floor(100/tpc))*chunk+1:end, :)];
% disp(test);
disp(train);
writetable(T,filename)
save([out,'_train_data_RF.txt'], 'train', '-ascii');
save([out,'_test_data_RF.txt'], 'test', '-ascii');

end