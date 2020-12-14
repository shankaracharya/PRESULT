function RF_train_test2(varargin)
fprintf('\n Example command line: "./presult train_test2 RF --train_data <train_data.txt> --test_data <test_data.txt>" \n')
p = inputParser;
addParameter(p,'train_data','pid_train_data.txt',@ischar);
addParameter(p,'test_data','pid_test_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'var','variable.txt',@ischar);
addParameter(p,'Hoeffding','yes',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'min_parent','100');
addParameter(p,'min_leaf','70');
addParameter(p,'n_trees','500');
addParameter(p,'out','RF_tt2',@ischar);
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
addParameter(p,'pvalue_logistic','0.05');
addParameter(p,'title','Random Forest',@ischar);

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
n_trees = (p.Results.n_trees);
out = (p.Results.out);
ran = (p.Results.randm);
log_reg = (p.Results.logistic_regression);
p_val_log = (p.Results.pvalue_logistic);
title_rf = (p.Results.title);
p_val = str2double(p_val_log);
min_parent = str2double(min_parent);
min_leaf = str2double(min_leaf);
n_trees = str2double(n_trees);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', train_file, '--output', 'RF_tt2_normalized_train_data.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', train_file, '--output', 'RF_tt2_normalized_train_data.txt', '--csv','--var',var);
end


if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', test_file, '--output', 'RF_tt2_normalized_test_data.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', test_file, '--output', 'RF_tt2_normalized_test_data.txt', '--csv','--var',var);
end

q = load('RF_tt2_normalized_train_data.txt');
test = load('RF_tt2_normalized_test_data.txt');

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
x = data_file(:,1:end-1);
t = data_file(:,end);
%%%%%%%%%%Start train_RF %%%%%%%%%%

okargs =   {'minparent' 'minleaf' 'nvartosample' 'ntrees' 'nsamtosample' 'method' 'oobe' 'weights'};
defaults = {min_parent min_leaf round(sqrt(size(x,2))) n_trees numel(t) 'r' 'n' []};
[~,~,minparent,minleaf,m,nTrees,n,method,oobe,W] = getargs(okargs,defaults,varargin{:});

avg_accuracy = 0;
for i = 1 : nTrees
     
    TDindx = round(numel(t)*rand(n,1)+.5);
    Random_ForestT = cartree(x(TDindx,:),t(TDindx), ...
        'minparent',minparent,'minleaf',minleaf,'method',method,'nvartosample',m,'weights',W);
    
    Random_ForestT.method = method;

    Random_ForestT.oobe = 1;
    if strcmpi(oobe,'y')        
        NTD = setdiff(1:numel(t),TDindx);
        tree_output = eval_cartree(x(NTD,:),Random_ForestT)';
        
        switch lower(method)        
            case {'c','g'}                
                Random_ForestT.oobe = numel(find(tree_output-Labels(NTD)'==0))/numel(NTD);
            case 'r'
                Random_ForestT.oobe = sum((tree_output-Label(NTD)').^2);
        end        
    end
    
    Random_Forest(i) = Random_ForestT;

    accuracy = Random_ForestT.oobe;
    if i == 1
        avg_accuracy = accuracy;
    else
        avg_accuracy = (avg_accuracy*(i-1)+accuracy)/i;
    end

end



%%%%%%%%%End train_RF %%%%%%%%%%%%%

save ([out,'_trained_net_tt2'],'Random_Forest');
xtest = test(:,1:end-1);
ttest = test(:,end);
y_RF_test = eval_RF(xtest,Random_Forest);
y_RF_train = eval_RF(x,Random_Forest);
save([out,'_test_score_tt2.txt'], 'y_RF_test', '-ascii');
save([out,'_train_score_tt2.txt'], 'y_RF_train', '-ascii');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[x_RF_train,y_RF_train,~,AUC_RF_train] = perfcurve(t,y_RF_train',size(t,2));
[C]=confmat(y_RF_test',ttest);
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

fprintf(' Sensitivity of the Test Model = %4.3f\r', Sensitivity);
fprintf('\n Specificity of the Test Model = %4.3f\r', Specificity);
fprintf('\n Accuracy of the Test Model = %4.3f\r', Accuracy);
[x_RF_test,y_RF_test,~,AUC_RF_test] = perfcurve(ttest,y_RF_test',size(ttest,2));
fprintf('\n AUC of the Train Model = %4.3f\r', AUC_RF_train);
fprintf('\n AUC of the Test Model = %4.3f\r', AUC_RF_test);


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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% LOGISTIC REGRESSION TEST%%%%%%%%%%%%
[x_logistic_train,y_logistic_train,~,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));
xxtest = xtest(:,ind_pv);
bb_train = b_train(ind_pv);
ax3_test = bsxfun(@times,bb_train',xxtest);
s2_test = sum(ax3_test,2);
[x_logistic_test,y_logistic_test,~,AUC_Logistic_test] = perfcurve(ttest,s2_test',size(ttest,2));
fprintf('\n AUC of the Logistic Train Model = %4.3f\r', AUC_Logistic_train);
fprintf('\n AUC of the Logistic Test Model = %4.3f\r', AUC_Logistic_test);
%%%%%%%%%%%%%%%

if strcmp(log_reg,'on');
   plot(x_RF_train,y_RF_train,x_RF_test,y_RF_test,x_logistic_train,y_logistic_train,x_logistic_test,y_logistic_test,'-');
   xlabel_tt2 = xlabel('1-Specificity');
   set(xlabel_tt2,'FontSize',16);
   ylabel_tt2 = ylabel('Sensitivity');
   set(ylabel_tt2,'FontSize',16);
   set(gcf,'color','w');
   set(gca,'FontSize',16);
   grid on;
   box on;
   title_tt2 = title(sprintf(title_rf));
   set(title_tt2,'FontSize',16);
   legend_RF_tt2 = legend(sprintf('train-RF =%4.3f',AUC_RF_train), sprintf('test-RF =%4.3f',AUC_RF_test), sprintf('train-log =%4.3f',AUC_Logistic_train), sprintf('test-log =%4.3f',AUC_Logistic_test),'Location','SouthEast');
   set(legend_RF_tt2,'FontSize',16);
   save([out,'_logistic_regression_model.txt'],'b_train_ori','-ascii');
   save([out,'_logistic_p_value.txt'],'p_value','-ascii');
   fprintf('\n logistic regression model file name: ');
   logistic_regression_model = [out,'_logistic_regression_model.txt'];
   disp(logistic_regression_model);
   fprintf('\n logistic regression model p_value file name: ');
   logistic_regression_pvalue = [out,'_logistic_p_value.txt'];
   disp(logistic_regression_pvalue);
else
   strcmp(log_reg,'off');
   plot(x_RF_train,y_RF_train,x_RF_test,y_RF_test,'-');
   xlabel_tt2 = xlabel('1-Specificity');
   set(xlabel_tt2,'FontSize',16);
   ylabel_tt2 = ylabel('Sensitivity');
   set(ylabel_tt2,'FontSize',16);
   grid on;
   box on;
   set(gcf,'color','w');
   set(gca,'FontSize',16);
   title_tt2 = title(sprintf(title_rf));
   set(title_tt2,'FontSize',16);
   legend_RF_tt2 = legend(['train-RF =',num2str(AUC_RF_train)], ['test-RF =',num2str(AUC_RF_test)],'Location','SouthEast');
   set(legend_RF_tt2,'FontSize',16);
end
graph = [out,'_train_test2'];
export_fig(graph, '-pdf', '-eps');
end