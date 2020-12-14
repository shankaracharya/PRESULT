function RT_train_test2(varargin)
% train_test2 RT --train_data pid_train_data.txt --test_data pid_test_data.txt --randm no --out Final_publication_result_tt2_RT_pid --min_parent 100 --min_leaf 40.
fprintf('\n Example command line: "./presult train_test2 RT --train_data <train_data.txt> --test_data <test_data.txt>" \n') 
p = inputParser;
addParameter(p,'train_data','pid_train_data.txt',@ischar);
addParameter(p,'test_data','pid_test_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'var','variable.txt',@ischar)
addParameter(p,'out','RT',@ischar);
addParameter(p,'min_parent','55');
addParameter(p,'min_leaf','50');
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
addParameter(p,'pvalue_logistic','0.01');
addParameter(p,'title','Regression Tree',@ischar);

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

new_array(strcmp('yes',new_array))={'shuffle'};
new_array(strcmp('no',new_array))={'default'};

parse(p,new_array{:});

train_file = (p.Results.train_data);
test_file = (p.Results.test_data);
csv = (p.Results.csv);
var = (p.Results.var);
miss = (p.Results.missing_value);
min_parent = (p.Results.min_parent);
min_leaf = (p.Results.min_leaf);
out = (p.Results.out);
ran = (p.Results.randm);
log_reg = (p.Results.logistic_regression);
p_val_log = (p.Results.pvalue_logistic);
title_rt = (p.Results.title);

min_parent = str2double(min_parent);
min_leaf = str2double(min_leaf);
p_val = str2double(p_val_log);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', train_file, '--output', 'normalized_train_data_tt2_RT.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', train_file, '--output', 'normalized_train_data_tt2_RT.txt', '--csv','--var',var);
end

q = load('normalized_train_data_tt2_RT.txt');


if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', test_file, '--output', 'normalized_test_data_tt2_RT.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', test_file, '--output', 'normalized_test_data_tt2_RT.txt', '--csv','--var',var);
end

%%%%%%%%%%%%%%%%%%%%
%DataSorting start
%%%%%%%%%%%%%%%%%%%%
rownumbers = size(q,1);
rng(ran);
ran = rand(rownumbers,1);
data_file2 = [ran q];
data_file3 = sortrows(data_file2,1);
data_file = data_file3(:,2:end);
%%%%%%%%%%%%%%%%%%%%%
ninput = size(data_file,2);
x = data_file(:,1:ninput-1);
t = data_file(:,ninput);
rtree = fitrtree(x,t,'MinParent',min_parent,'MinLeaf', min_leaf);
save ([out,'_trained_net_tt2'],'rtree');
test = load('normalized_test_data_tt2_RT.txt');
ninput = size(test,2);
xtest = test(:,1:ninput-1);
ttest = test(:,ninput);
ytest = predict(rtree, xtest);
ytrain = predict(rtree, x);
save([out,'_test_score_tt2.txt'], 'ytest', '-ascii');
save([out,'_train_score_tt2.txt'], 'ytrain', '-ascii');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x_RT_train,y_RT_train,~,AUC_RT_train] = perfcurve(t,ytrain',size(t,2));
[C]=confmat(ytest,ttest);
fprintf('\n Confusion Matrix of the Test Model: \r\n');
disp(C);
C1 = C(1,1);
C2 = C(1,2);
C3 = C(2,1);
C4 = C(2,2);
Correct_prediction = C1+C4;
Wrong_prediction = C2+C3;
Accuracy = (Correct_prediction/(Correct_prediction + Wrong_prediction))*100;
Sensitivity = C1/(C1+C3);
Specificity = C4/(C4+C2);


fprintf(' Sensitivity of the RT Test Model = %4.3f\r', Sensitivity);
fprintf('\n Specificity of the RT Test Model = %4.3f\r', Specificity);
fprintf('\n Accuracy of the RT Test Model = %4.3f\r', Accuracy);
[x_RT_test,y_RT_test,~,AUC_RT_test] = perfcurve(ttest,ytest',size(ttest,2));
fprintf('\n AUC of the RT Train Model = %4.3f\r', AUC_RT_train);
fprintf('\n AUC of the RT Test Model = %4.3f\r', AUC_RT_test);


%%%%%%%%% LOGISTIC REGRESSION TRAIN%%%%%%%%%%%%
[b_train_ori,~,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train_ori(2:end);
p_value = stats.p;
stats.p = stats.p(2:end);
ind_pv = stats.p<=p_val;
x_train = x(:,ind_pv);
b_train_pv = b_train(ind_pv);
ax3_train = bsxfun(@times,b_train_pv',x_train);
s2_train = sum(ax3_train,2);
%%%%%%%% Logistic regression %%%%%%%%%%

%%%%%%%%% LOGISTIC REGRESSION TEST%%%%%%%%%%%%
[x_logistic_train,y_logistic_train,~,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));
xxtest = xtest(:,ind_pv);
bb_train = b_train(ind_pv);
ax3_test = bsxfun(@times,bb_train',xxtest);
s2_test = sum(ax3_test,2);
[x_logistic_test,y_logistic_test,~,AUC_Logistic_test] = perfcurve(ttest,s2_test',size(ttest,2));


%%%%%%%%%%%%%%%

if strcmp(log_reg,'on');
    plot(x_RT_train,y_RT_train,x_RT_test,y_RT_test,x_logistic_train,y_logistic_train,x_logistic_test,y_logistic_test,'-');
    xlabel_tt2 = xlabel('1-Specificity');
    set(xlabel_tt2,'FontSize',16);
    ylabel_tt2 = ylabel('Sensitivity');
    set(ylabel_tt2,'FontSize',16);
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    grid on;
    title_tt2 = title(sprintf(title_rt));
    set(title_tt2,'FontSize',16)
    legend_RT_tt2 = legend(sprintf('train-RT =%4.3f',AUC_RT_train), sprintf('test-RT =%4.3f',AUC_RT_test), sprintf('train-log =%4.3f',AUC_Logistic_train), sprintf('test-log =%4.3f',AUC_Logistic_test),'Location','SouthEast');
    set(legend_RT_tt2,'FontSize',16);
    fprintf('\n AUC of the Logistic Train Model = %4.3f\r', AUC_Logistic_train);
    fprintf('\n AUC of the Logistic Test Model = %4.3f\r', AUC_Logistic_test);
    graph = [out,'_train_test2'];
    export_fig(graph, '-pdf', '-eps');
    save([out,'_logistic_regression_model.txt'],'b_train_ori','-ascii');
    save([out,'_logistic_p_value.txt'],'p_value','-ascii');
    fprintf('\n logistic regression model file name: ');
    logistic_regression_model = [out,'_logistic_regression_model.txt'];
    disp(logistic_regression_model);
    fprintf('\n logistic regression model p_value file name: ');
    logistic_regression_pvalue = [out,'_logistic_p_value.txt'];
    disp(logistic_regression_pvalue);
else
    plot(x_RT_train,y_RT_train,x_RT_test,y_RT_test,'-');
    xlabel_tt2 = xlabel('1-Specificity');
    set(xlabel_tt2,'FontSize',16);
    ylabel_tt2 = ylabel('Sensitivity');
    set(ylabel_tt2,'FontSize',16);
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    grid on;
    title_tt2 = title(sprintf(title_rt));
    set(title_tt2,'FontSize',16)
    legend_RT_tt2 = legend(sprintf('train-RT =%4.3f',AUC_RT_train), sprintf('test-RT =%4.3f',AUC_RT_test),'Location','SouthEast');
    set(legend_RT_tt2,'FontSize',16)
end
graph = [out,'_train_test2'];
export_fig(graph, '-pdf', '-eps');
end