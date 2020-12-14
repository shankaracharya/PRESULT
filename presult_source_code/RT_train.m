function RT_train(varargin)
fprintf('\n Example command line: "./presult train RT --train_data <training_data.txt>" \n');
p = inputParser;
addParameter(p,'train_data','pid_train_data.txt',@ischar);
addParameter(p,'csv','no',@ischar);
addParameter(p,'var','variable.txt',@ischar)
addParameter(p,'missing_value','?');
addParameter(p,'min_parent','50');
addParameter(p,'out','RT',@ischar);
addParameter(p,'min_leaf','12');
addParameter(p,'randm','shuffle',@ischar);
addParameter(p,'logistic_regression','on',@ischar);
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
out = (p.Results.out);
csv = (p.Results.csv);
var = (p.Results.var);
miss = (p.Results.missing_value);
ran = (p.Results.randm);
log_reg = (p.Results.logistic_regression);
min_parent = (p.Results.min_parent);
min_leaf = (p.Results.min_leaf);
title_rt = (p.Results.title);
min_parent = str2double(min_parent);
min_leaf = str2double(min_leaf);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', train_file, '--output', 'RT_normalized_train_data.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', train_file, '--output', 'RT_normalized_train_data.txt', '--csv','--var',var);
end

q = load('RT_normalized_train_data.txt');


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

ninput = size(data_file,2);
x = data_file(:,1:ninput-1);
t = data_file(:,ninput);
rtree = fitrtree(x,t,'MinParent',min_parent,'MinLeaf', min_leaf);
y_RT_train = predict(rtree,x);
save ([out,'_trained_net'],'rtree');
save([out,'_train_score.txt'], 'y_RT_train', '-ascii');
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

[x_train,y_train,~,AUC_RT_train] = perfcurve(t, y_RT_train',size(t,2));
[x_logistic_train,y_logistic_train,~,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));

if strcmp(log_reg,'on');
    fprintf('\n AUC of the Train Model for RT = %4.3f\r', AUC_RT_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %4.3f\r', AUC_Logistic_train);
    plot(x_train,y_train,'m-', x_logistic_train,y_logistic_train, 'b-');
    box on;
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    grid on;
    title_t = title(sprintf(title_rt));
    set(title_t,'FontSize',16);
    legend_RT_Train = legend(sprintf('AUC-RT = %4.2f',AUC_RT_train), sprintf('AUC-LOG = %4.2f',AUC_Logistic_train),'Location','SouthEast');
    set(legend_RT_Train,'FontSize',16);
    fprintf('\n logistic regression model file name: ');
    logistic_regression_model = [out,'_logistic_regression_model.txt'];
    disp(logistic_regression_model);
    save([out,'_logistic_regression_model.txt'],'b_train','-ascii');
    fprintf('\n logistic regression model p_value file name: ');
    logistic_regression_pvalue = [out,'_logistic_p_value.txt'];
    disp(logistic_regression_pvalue);
    save([out,'_logistic_p_value.txt'],'pval','-ascii');
else
    fprintf('\n AUC of the Train Model for RT = %4.3f\r\n', AUC_RT_train);
    fprintf('\n AUC of the Train Model for Logistic regression = %4.3f\r\n', AUC_Logistic_train);
    strcmp(log_reg,'off');
    plot(x_train,y_train, 'm-');
    xlabel_train = xlabel('1-Specificity');
    ylabel_train = ylabel('Sensitivity');
    set(xlabel_train,'FontSize',16);
    set(ylabel_train,'FontSize',16);
    box on;
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    grid on;
    legend_RT_Train = legend(sprintf('AUC-RT = %4.2f',AUC_RT_train),'Location','SouthEast');
    set(legend_RT_Train,'FontSize',16);
    title_t = title(sprintf(title_rt));
    set(title_t,'FontSize',16);
end
output_train = [out, '_train_ROC'];
export_fig (output_train, '-pdf', '-eps');
end