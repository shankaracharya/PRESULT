function RF_kfold(varargin)
fprintf('\n Example command line: "./presult kfold RF --input_data <input_data.txt> --randm <yes/no> --kfold <int>" \n')
p = inputParser;
addParameter(p,'input_data','pid_raw_data.txt',@ischar);
addParameter(p,'kfold','10');
addParameter(p,'csv','no',@ischar);
addParameter(p,'var','variable.txt',@ischar)
addParameter(p,'missing_value','?');
addParameter(p,'out','RF',@ischar);
addParameter(p,'min_parent','100');
addParameter(p,'min_leaf','15');
addParameter(p,'n_trees','300');
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
data_file = (p.Results.input_data);
csv = (p.Results.csv);
var = (p.Results.var);
miss = (p.Results.missing_value);
kvalue = (p.Results.kfold);
min_parent = (p.Results.min_parent);
min_leaf = (p.Results.min_leaf);
n_trees = (p.Results.n_trees);
out = (p.Results.out);
ran = (p.Results.randm);
title_rf = (p.Results.title);

if strcmp(csv,'no') && strcmp(miss,'?') 
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_kfold_RF.txt','--var',var);
else
   perl('newest_normalization.pl', '--input', data_file, '--output', 'normalized_data_kfold_RF.txt', '--csv','--var',var);
end
data_file1 = load('normalized_data_kfold_RF.txt');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = str2double(kvalue);
min_parent = str2double(min_parent);
min_leaf = str2double(min_leaf);
n_trees = str2double(n_trees);
rownumbers = size(data_file1,1);
%%%%%%%%%%%%%%%%%%%%
%DataSorting start
%%%%%%%%%%%%%%%%%%%%
rng(ran);
ran = rand(rownumbers,1);
data_file2 = [ran data_file1];
data_file3 = sortrows(data_file2,1);
data_file = data_file3(:,2:end);
%%%%%%%%%%%%%%%%%%%%%
%Data Sorting end
%%%%%%%%%%%%%%%%%%%%%%
chunk = floor(rownumbers/k);
hold all;
grid on;
ah1 = gca;
for f = 1:k
    fprintf('\n ############ %d-fold cross validation ############### \n',f);
    test = data_file((f-1)*chunk+1:(f)*chunk,:);
    train = [data_file(1:(f-1)*chunk,:); data_file(f*chunk+1:end, :)];
    save ([out,'_train_data_kfold_',int2str(f),'.txt'],'train','-ascii');
    save ([out,'_test_data_kfold_',int2str(f),'.txt'],'test','-ascii');
    fprintf(['\n Train data for ', int2str(f), '-kfold ', 'is saved as: ']);
    fprintf([out,'_train_data_kfold_',int2str(f),'.txt'],'\n');
    fprintf('\n');
    fprintf(['\n Test data for ', int2str(f), '-kfold ', 'is saved as: ']);
    fprintf([out,'_test_data_kfold_',int2str(f),'.txt'],'\n');
    fprintf('\n');
    q = load([out,'_train_data_kfold_',int2str(f),'.txt']);
    ninput = size(q,2);
    x = q(:,1:ninput-1);
    t = q(:,ninput);

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
    save ([out,'_trained_net_kfold_',int2str(f)],'Random_Forest');
    fprintf(['\n Trained network for ', int2str(f), '-kfold ', 'RF Model is saved as: ']);
    fprintf([out,'_trained_net_kfold_',int2str(f),'.mat'],'\n');
    fprintf('\n');
    test = load([out,'_test_data_kfold_',int2str(f),'.txt']);
    ninput = size(test,2);
    xtest = test(:,1:ninput-1);
    ttest = test(:,ninput);
    y_test = eval_RF(xtest,Random_Forest);
%     disp(y_test);
    [C_RF]=confmat(y_test',ttest);
    fprintf('\n Confusion Matrix of the Test Model: \r\n');
    disp(C_RF);
    C1 = C_RF(1,1);
    C2 = C_RF(1,2);
    C3 = C_RF(2,1);
    C4 = C_RF(2,2);
    Correct_prediction = C1+C4;
    Wrong_prediction = C2+C3;
    Sensitivity = C1/(C1+C3);
    Specificity = C4/(C4+C2);
    Accuracy = (Correct_prediction/(Correct_prediction + Wrong_prediction))*100;
    fprintf('\n Sensitivity of the RF Model = %f\r', Sensitivity);
    fprintf('\n Specificity of the RF Model = %f\r', Specificity);
    fprintf('\n Accuracy of the RF Model = %f\r', Accuracy);
    [faRate, hitRate, AUC_test] = rocCurve(y_test, ttest);
    fprintf('\n AUC of the RF Model = %f\r\n', AUC_test);
    
    legendinfo{f} = ['AUC = ' num2str(AUC_test)];
    AUC_COL(f,:) = AUC_test;
    ACCU(f,:) = Accuracy;
    SENS(f,:) = Sensitivity;
    SPEC(f,:) = Specificity;
    kfold(f,:) = f;
               
end
    hold off;
    Avg_AUC_test = mean(AUC_COL);
    legend(ah1, (legendinfo),'Location','SouthEast','FontSize',14);
    xlabel_kf = xlabel(ah1,'1-Specificity');
    set(xlabel_kf,'FontSize',16);
    ylabel_kf = ylabel(ah1, 'Sensitivity');
    set(ylabel_kf,'FontSize',16);
    set(gcf,'color','w');
    set(gca,'FontSize',16);
    box on;
    grid on;
    dim =[0.63 0.4 0.1 0.2];
    str = ['Avg AUC = ',num2str(Avg_AUC_test)];
    annotation('textbox',dim,'String',str,'FitBoxToText','on','FontSize',14,'Color','k');
    Average_sesitivity = mean(SENS);
    Average_specificity = mean(SPEC);
    Average_accuracy = mean(ACCU);
    fprintf('\n ############ Summary for %d-fold cross validation ############### \n',k);
    summary = [kfold SENS SPEC ACCU AUC_COL];
    TT = array2table(summary,'VariableNames',{'kfold' 'Sensitivity' 'Specificity' 'Accuracy' 'AUC_test'});
    disp(TT);
    fprintf('\n Average Accuracy of the Test Model = %f\r\n', Average_accuracy);
    Avg_AUC = mean(AUC_COL);
    title_kf = title(ah1, sprintf(title_rf));
    set(title_kf,'FontSize',16);
    fprintf('\n Average Sensitivity of the Test Model = %f\r\n', Average_sesitivity);
    fprintf('\n Average Specificity of the Test Model = %f\r\n', Average_specificity);
    fprintf('\n Average AUC of the Test Model = %f\r\n', Avg_AUC);
    kfold_out = [out,'_kfold_ROC'];
    export_fig (kfold_out, '-pdf', '-eps');
 
end