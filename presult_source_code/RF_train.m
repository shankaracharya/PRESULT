function RF_train(varargin)
fprintf('\n Example command line: "./presult train RF --train_data <training_data.txt>" \n');
p = inputParser;
addParameter(p,'train_data','pid_train_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'var','variable.txt',@ischar);
addParameter(p,'out','RF',@ischar);
addParameter(p,'n_trees','300');
addParameter(p,'min_parent','100');
addParameter(p,'min_leaf','70');
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
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
n_trees = (p.Results.n_trees);
out = (p.Results.out);
min_parent = (p.Results.min_parent);
min_leaf = (p.Results.min_leaf);
var = (p.Results.var);
ran = (p.Results.randm);
log_reg = (p.Results.logistic_regression);
csv = (p.Results.csv);
miss = (p.Results.missing_value);
title_rf = (p.Results.title);
n_trees = str2double(n_trees);
min_leaf = str2double(min_leaf);
min_parent = str2double(min_parent);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', train_file, '--output', 'RF_normalized_train_data.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', train_file, '--output', 'RF_normalized_train_data.txt', '--csv','--var',var);
end

q = load('RF_normalized_train_data.txt');
rownumbers = size(q,1);

%%%%%%%%%%%%%%%%%%%%
%DataSorting start
%%%%%%%%%%%%%%%%%%%%
rng(ran);
ran = rand(rownumbers,1);
data_file2 = [ran q];
data_file3 = sortrows(data_file2,1);
data_file = data_file3(:,2:end);
%%%%%%%%%%%%%%%%%%%%%
%Data Sorting end
%%%%%%%%%%%%%%%%%%%%%%
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

save ([out,'_trained_net'],'Random_Forest');
y_RF_train = eval_RF(x,Random_Forest);
save([out,'_train_score.txt'], 'y_RF_train', '-ascii');
%%%%%%%%% LOGISTIC REGRESSION %%%%%%%%%%%%
[b_train_ori,~,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train_ori(2:end);
pval = stats.p;
p_vals = stats.p(2:end);
ind_pv = p_vals<=0.01;
b_train_pv = b_train(ind_pv);
x_train_pv = x(:,ind_pv);
ax3_train = bsxfun(@times,b_train_pv',x_train_pv);
s2_train = sum(ax3_train,2);


%%%%%%%% Logistic regression %%%%%%%%%%

[x_train,y_train,~,AUC_RF_train] = perfcurve(t, y_RF_train',size(t,2));
[x_logistic_train,y_logistic_train,~,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));

if strcmp(log_reg,'on')
    fprintf('\n AUC of the Train Model for RF = %4.2f\r', AUC_RF_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %4.2f\r', AUC_Logistic_train);
    plot(x_train,y_train,'m-',x_logistic_train,y_logistic_train, 'b-');
    hold off;
    box on;
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    grid on;
    title_t = title(sprintf(title_rf));
    set(title_t,'FontSize',16);
    legend_RF_Train = legend([sprintf('AUC-RF = %4.2f',AUC_RF_train)], [sprintf('AUC-LOG = %4.2f',AUC_Logistic_train)],'Location','SouthEast'); %#ok<*NBRAK>
    set(legend_RF_Train,'FontSize',16);
    fprintf('\n logistic regression model file name: ');
    logistic_regression_model = [out,'_logistic_regression_model.txt'];
    disp(logistic_regression_model);
    save([out,'_logistic_regression_model.txt'],'b_train_ori','-ascii');
    fprintf('\n logistic regression model p_value file name: ');
    logistic_regression_pvalue = [out,'_logistic_p_value.txt'];
    disp(logistic_regression_pvalue);
    save([out,'_logistic_p_value.txt'],'pval','-ascii');
else
    fprintf('\n AUC of the Train Model for RF = %4.2f\r', AUC_RF_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %4.2f\r', AUC_Logistic_train);
    strcmp(log_reg,'off');
    plot(x_train,y_train, 'b-');
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    box on;
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    grid on;
    title_t = title(sprintf(title_rf));
    set(title_t,'FontSize',16);
    legend_RF_Train = legend([sprintf('AUC-RF = %4.2f',AUC_RF_train)], 'Location','SouthEast'); 
    set(legend_RF_Train,'FontSize',16);
end
output_train = [out, '_train_ROC'];
export_fig (output_train, '-pdf', '-eps');
end