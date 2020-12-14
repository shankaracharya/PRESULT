function RF_train_abs_risk(varargin)
fprintf('\n Example command line: "./presult train_abs_risk RF --train_data <train_data.txt> --randm <yes/no> --id_age <id_age_label.txt>" \n');
p = inputParser;
addParameter(p,'train_data','pid_raw_data_with_id.txt',@ischar);
addParameter(p,'id_age','pid_id_age_label.txt')
addParameter(p,'csv','no',@ischar);
addParameter(p,'missing_value','?');
addParameter(p,'var','variable.txt',@ischar);
addParameter(p,'out','RF_risk',@ischar);
addParameter(p,'n_trees','300');
addParameter(p,'min_parent','100');
addParameter(p,'min_leaf','70');
addParameter(p,'randm','shuffle',@ischar);
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
id_age = (p.Results.id_age);
n_trees = (p.Results.n_trees);
out = (p.Results.out);
min_parent = (p.Results.min_parent);
min_leaf = (p.Results.min_leaf);
var = (p.Results.var);
ran = (p.Results.randm);
csv = (p.Results.csv);
miss = (p.Results.missing_value);
title_rf = (p.Results.title);
n_trees = str2double(n_trees);
min_leaf = str2double(min_leaf);
min_parent = str2double(min_parent);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', train_file, '--output', 'normalized_train_data_RF.txt','--var',var,'--with_id');

else
   perl('newest_normalization.pl', '--input', train_file, '--output', 'normalized_train_data_RF.txt', '--csv','--var',var,'--with_id');
end

q = load('normalized_train_data_RF.txt');


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
x = data_file(:,2:end-1);
save([out,'_training_data_RF.txt'],'x','-ascii')
t = data_file(:,end);
save([out,'_outcome.txt'],'t','-ascii');

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
save ([out,'_trained_net'],'Random_Forest');
fprintf('\n The RF model saved as: ');
fprintf(out);fprintf('_trained_net.mat \n')
y_train = eval_RF(x,Random_Forest);
y_train = y_train';
log_odd_ratio = log((y_train)./(1-y_train));
[b_one,dev,stats] = glmfit((log_odd_ratio),t,'binomial','link','logit');
% b_zero = b_one(1);
b_one = b_one(2);
fprintf('\n beta one value: %f\r\n', b_one);
% fprintf('\n beta zero value: %f\r\n', b_zero);
% save([out,'_training_score.txt'],'y_train','-ascii');
%%%%%%%%% LOGISTIC REGRESSION Train%%%%%%%%%%%%
[b_train,dev,stats] = glmfit(x,t,'binomial','link','logit');
b_train = b_train(2:end);% disp(t);
stats.p = stats.p(2:end);
ind_pv = stats.p<=0.01;
x_train = x(:,ind_pv);
b_train_pv = b_train(ind_pv);
ax3_train = bsxfun(@times,b_train_pv',x_train);
s2_train = sum(ax3_train,2);
save([out,'_log_train_out.txt'],'b_train','-ascii')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Calculation of r_second_term %%%%%%%%%%
ind_affected = find(t==1);
three_percentile = prctile(x,[25 50 75]);
bb_50th_per = three_percentile(2,2:end);
% disp(bb_50th_per);
fprintf(' The median baseline risk values without age saved as: ');
fprintf(out);fprintf('_median_baseline.txt \n');
fprintf('\n');
save([out,'_median_baseline.txt'],'bb_50th_per','-ascii');
fprintf(' median_baseline risk values without age: ');
disp(bb_50th_per);
%%%%%%%%%%%%%%%%% Calculation of r_first_term %%%%%%%%%%
id_age_label_data = load(id_age);
ind_id_age_affected = id_age_label_data(:,3)==1;
all_affected_Age = id_age_label_data(ind_id_age_affected,:);
affected_Age = all_affected_Age(:,2);
affected_y_train = y_train(ind_affected);
size_affected_y_train = size(affected_y_train,1);
fprintf(' number of cases in train dataset: %d\r', size_affected_y_train);
affected_log_odd_ratio = log((affected_y_train)./(1-affected_y_train));
rj_first_term = b_one.*affected_log_odd_ratio;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%Calculation of r_second_term for 50th percentile %%%%%%%%%%%%%%
r_j_cases_matrix_50th_per = [affected_Age(:,1), repmat(bb_50th_per,size(affected_Age,1),1)];
test_case_matrix_50th_per = r_j_cases_matrix_50th_per(:,1:end);
% save([out,'_test_case_matrix_50th_per.txt'],'test_case_matrix_50th_per','-ascii');
y_50th_perc = eval_RF(test_case_matrix_50th_per,Random_Forest);
y_50th_per = y_50th_perc';
affected_log_odd_ratio_50th_perc = (log((y_50th_per)./(1-y_50th_per)));
rj_second_term_50th_perc = b_one.*affected_log_odd_ratio_50th_perc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%calculation of rj and Attributable Risk %%%%%%%%%%%%%%%%%%

rj_50th_per = exp(rj_first_term)./exp(rj_second_term_50th_perc);
sum_of_one_by_rj_50th = sum(1./rj_50th_per);
AR_50th = 1-((1/size_affected_y_train).*sum_of_one_by_rj_50th);
fprintf('\n Attributable risk value: %f\r', AR_50th);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[x_train,y_train,T_train,AUC_RF_train] = perfcurve(t, y_train',size(t,2));
[x_logistic_train,y_logistic_train,T_logistic_train,AUC_Logistic_train] = perfcurve(t,s2_train',size(t,2));
fprintf('\n AUC of the Train Model for RF = %f\r', AUC_RF_train);
plot(x_train,y_train, x_logistic_train, y_logistic_train,'-');
box on;
set(gcf,'color','w');
set(gca,'FontSize',16);
xlabel_train = xlabel('1-Specificity');
ylabel_train = ylabel('Sensitivity');
set(xlabel_train,'FontSize',18);
set(ylabel_train,'FontSize',18);
grid on;
title_t = title(sprintf(title_rf));
set(title_t,'FontSize',16);
legend_tab_risk = legend(['AUC-RF = ',num2str(AUC_RF_train)], ['AUC-LOG = ',num2str(AUC_Logistic_train)],'Location','SouthEast');
set(legend_tab_risk,'FontSize',16);
output_train = [out, '_train_ROC'];
export_fig (output_train, '-pdf', '-eps');
end