function RF_train_test1(varargin)
fprintf('\n Example command line: "./presult train_test1 RF --input_data <input_data.txt>" \n');
p = inputParser;
addParameter(p,'input_data','pid_raw_data.txt',@ischar);
addParameter(p,'out','RF',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'var','variable.txt',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'test_pc','20');
addParameter(p,'n_trees','300');
addParameter(p,'min_parent','176');
addParameter(p,'min_leaf','15');
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
addParameter(p,'pvalue_logistic','0.01');
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


data_file = (p.Results.input_data);

out = (p.Results.out);
tpc = (p.Results.test_pc);
csv = (p.Results.csv);
var = (p.Results.var);
miss = (p.Results.missing_value);
ran = (p.Results.randm);
n_trees = (p.Results.n_trees);
min_parent = (p.Results.min_parent);
min_leaf = (p.Results.min_leaf);
log_reg = (p.Results.logistic_regression);
title_rf = (p.Results.title);
p_val_log = (p.Results.pvalue_logistic);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_tt1_RF.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_tt1_RF.txt', '--csv','--var',var);
end

data_file1 = load('normalized_data_tt1_RF.txt');

tpc = str2double(tpc);
n_trees = str2double(n_trees);
min_leaf = str2double(min_leaf);
min_parent = str2double(min_parent);
p_val = str2double(p_val_log);

rownumbers = size(data_file1,1);
%%%%%%%%%%%%%%%%%%%%
%DataSorting start
%%%%%%%%%%%%%%%%%%%
rng(ran);
ran = rand(rownumbers,1);
data_file2 = [ran data_file1];
data_file3 = sortrows(data_file2,1);
data_file = data_file3(:,2:end);
%%%%%%%%%%%%%%%%%%%%%
%Data Sorting end
%%%%%%%%%%%%%%%%%%%%%%
chunk = floor(rownumbers*tpc/100);
test = data_file(((floor(100/tpc)-1))*chunk+1:(floor(100/tpc))*chunk,:);
train = [data_file(1:((floor(100/tpc))-1)*chunk,:); data_file((floor(100/tpc))*chunk+1:end, :)];
save([out,'_train_data.txt'], 'train', '-ascii');
save([out,'_test_data.txt'], 'test', '-ascii');
train_input = [out,'_train_data.txt'];
test_input = [out,'_test_data.txt'];
q = load(train_input);
ninput = size(q,2);
x = q(:,1:ninput-1);
t = q(:,ninput);

%%%%%%%RANDOM FOREST %%%%%%%%%%
% Random_Forest=train_RF(x,t);

%%%%%%%%%%Start train_RF %%%%%%%%%%

okargs =   {'minparent' 'minleaf' 'nvartosample' 'ntrees' 'nsamtosample' 'method' 'oobe' 'weights'};
defaults = {min_parent min_leaf round(sqrt(size(x,2))) n_trees numel(t) 'r' 'n' []};
[eid,emsg,minparent,minleaf,m,nTrees,n,method,oobe,W] = getargs(okargs,defaults,varargin{:});

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

save ([out,'_trained_net_tt1'],'Random_Forest');
test = load(test_input);
ninput = size(test,2);
xtest = test(:,1:ninput-1);
ttest = test(:,ninput);
y_test = eval_RF(xtest,Random_Forest);
y_train = eval_RF(x,Random_Forest);
save([out,'_test_score_tt1.txt'], 'y_test', '-ascii');
save([out,'_train_score_tt1.txt'], 'y_train', '-ascii');
% save([out,'_ttest.txt'], 'ttest', '-ascii');
% save([out,'t.txt'], 't', '-ascii');
% save('Random_Forest','Random_Forest');
[C_RF]=confmat(y_test',ttest);
fprintf('\n Confusion Matrix of the Test Model: \r\n');
disp(C_RF);
C1 = C_RF(1,1);
C2 = C_RF(1,2);
C3 = C_RF(2,1);
C4 = C_RF(2,2);
Correct_prediction_RF = C1+C4;
Wrong_prediction_RF = C2+C3;
Accuracy = (Correct_prediction_RF/(Correct_prediction_RF + Wrong_prediction_RF))*100;
Sensitivity = C1/(C1+C3);
Specificity = C4/(C4+C2);
fprintf(' Sensitivity of the RF Test Model = %4.2f\r', Sensitivity);
fprintf('\n Specificity of the RF Test Model = %4.2f\r', Specificity);
fprintf('\n Accuracy of the RF Test Model = %4.2f\r', Accuracy);

%%%%%%%LOGISTIC_REGRESSION_TRAIN%%%%%%%%%%%%%%%%
[b_train_ori,~,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train_ori(2:end);
p_value = stats.p;
stats.p = stats.p(2:end);
ind_pv = stats.p<=p_val;
x_train_pv = x(:,ind_pv);
bb_train = b_train(ind_pv);
ax3_train = bsxfun(@times,bb_train',x_train_pv);
s2_train = sum(ax3_train,2);

%%%%%%%%LOGISTIC_REGRESSION_TEST%%%%%%%%%%%%%%%%
[x_logistic_train,y_logistic_train,~,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));
xxtest = xtest(:,ind_pv);
ax3_test = bsxfun(@times,bb_train',xxtest);
s2_test = sum(ax3_test,2);
[x_logistic_test,y_logistic_test,~,AUC_Logistic_test] = perfcurve(ttest,s2_test',size(ttest,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x_train,y_train,~,AUC_RF_train] = perfcurve(t, y_train',size(t,2));
[x_test,y_test,~,AUC_RF_test] = perfcurve(ttest,y_test',size(ttest,2));
fprintf('\n AUC of the RF Train Model = %4.2f\r', AUC_RF_train);        
fprintf('\n AUC of the RF Test Model = %4.2f\r', AUC_RF_test);
fprintf('\n AUC of the Logistic Test Model = %4.2f\r', AUC_Logistic_test);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(log_reg,'on')
   plot(x_train,y_train,'b-',x_test,y_test,'m-');
   hold all;
   set(gcf,'color','w');
   set(gca,'FontSize',16);
   box on;
   title_tt1 = title(sprintf(title_rf));
   set(title_tt1,'FontSize',16);
   plot(x_logistic_train,y_logistic_train,'r-',x_logistic_test,y_logistic_test,'g-');
   xlabel_tt1 = xlabel('1-Specificity');
   set(xlabel_tt1,'FontSize',16);
   ylabel_tt1 = ylabel('Sensitivity');
   set(ylabel_tt1,'FontSize',16);   
   grid on;
   legend_RF_tt1 = legend(sprintf('train-RF =%4.2f',AUC_RF_train), sprintf('test-RF =%4.2f',AUC_RF_test), sprintf('train-log = %4.2f',AUC_Logistic_train), sprintf('test-log =%4.2f',AUC_Logistic_test) ,'Location','SouthEast');
   set(legend_RF_tt1,'FontSize',16);
   save([out,'_logistic_regression_model.txt'],'b_train_ori','-ascii');
   save([out,'_logistic_p_value.txt'],'p_value','-ascii');
   fprintf('\n logistic regression model file name: ');
   logistic_regression_model = [out,'_logistic_regression_model.txt'];
   disp(logistic_regression_model);
   fprintf('\n logistic regression model p_value file name: ');
   logistic_regression_pvalue = [out,'_logistic_p_value.txt'];
   disp(logistic_regression_pvalue);
else
   plot(x_train,y_train,'b-',x_test,y_test,'m-');
   set(gcf,'color','w');
   set(gca,'FontSize',16);
   box on;
   xlabel_tt1 = xlabel('1-Specificity');
   set(xlabel_tt1,'FontSize',16);
   ylabel_tt1 = ylabel('Sensitivity');
   set(ylabel_tt1,'FontSize',16);
   grid on;
   title(sprintf(title_rf));
   legend_RF_tt1 = legend(sprintf('train-RF =%4.2f',AUC_RF_train), sprintf('test-RF =%4.2f',AUC_RF_test), 'Location','SouthEast');
   set(legend_RF_tt1,'FontSize',16);
end
output_tt1 = [out,'_train_test1'];
export_fig(output_tt1, '-eps', '-pdf');
end
