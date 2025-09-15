classdef util
    %UTIL Utility methods for ML data preparation workflows.
    %
    % util methods:
    %
    %   Format Conversion
    %   sequence2frame      - Convert variables to buffer/frame by optional grouping
    %   sequence2ensemble   - Convert table to ensemble form
    %   ensemble2sequence   - Convert ensemble to table form
    %   text2categorical    - Convert text variables to categorical
    %
    %   Feature Scaling
    %   normalize           - Scale data [zscore | range | auto]
    %   scaler              - Apply scaling params used in training to new data
    %
    %
    %   Feature Transform
    %   log10               - Apply logarithm base 10
    %   pow                 - Apply exponentiation (default 2)
    %   skewness            - Report skewness
    %   
    %
    %   Feature Engineering
    %   descriptivestatistics - Calculate descriptive statistics on buffer/frames
    %
    %
    %   Feature Selection
    %   rmconstant           - Remove constant variables
    %   rminterclassconstant - Remove variables with constants within classes
    %   fscchi2              - Univariate Chi-2 feature selection for classification
    %   fscmrmr              - Rank features for classification using minimum redundancy maximum relevance (MRMR) algorithm
    %   fscanova             - Compute ranking score using One-Way ANOVA.
    %   
    %
    %   Dimension Reduction
    %   pca                 - Principal component dimension reduction
    %   cmds                - Classic multidimensional scaling dimension reduction
    %
    %
    %   Management
    %   summarize           - Flag missing data, constant vars, and constant 'in-class'
    %   name                - Return table variable names
    %   custom              - Return table custom properties
    %   isnormal            - True for normal distribution
    %   isnumeric           - True for numeric features
    %   iscategorical       - True for categorical features 
    %   isconstant          - True for constant features
    %   isvar               - True if specified names are in table
    %   istext              - True for text features (string,cellstr, or char)
    
    % Copyright 2021 The MathWorks Inc.
    
    methods (Static)
        
        function value = summarize( tbl, options )
            %SUMMARIZE Summarize content of table for ML.
            %
            % Syntax:
            %   value = util.summarize( tbl ) screens tbl for missing data
            %   and constant variables
            %   value = util.summarize( tbl, "Response", "name" ) screens
            %   tbl for additional criteria 'constantInClass' (e.g. if a
            %   variable if constant within a given reponse class)
            %
            % value is summary table with criteria: tF_Missing, tF_Constant,
            % and tF_ConstantInClass. Each criteria will report as
            % true if found, false if not, or NaN if not applicable.
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.Response (1,1) string {mustBeInTable(tbl,options.Response)} = ""
            end %arguments
            
            tF_Missing  = any( ismissing( tbl ), 'all' );
            tF_Constant = any( util.isconstant( tbl ) );
            
            if options.Response ~= ""
                tF_ConstantInClass  = any( util.isconstant( tbl, "Response", options.Response ), 'all' );
            else
                tF_ConstantInClass = NaN;
            end
            
            tF_CategoricalNotConvert = any( util.istext( tbl ) );
            
            value = table( tF_Missing, tF_Constant, tF_ConstantInClass, tF_CategoricalNotConvert );
            
        end %function
        
        
        function [value, info] = normalize( tbl, mthd, options )
            %NORMALIZE Scale specified features in table for ML
            %
            % Syntax:
            %   [value, info] = normalize( tbl )
            %   [value, info] = normalize( tbl, mthd, ...
            %       "DataVariables", varnames )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                mthd {mustBeMember(mthd, ...
                    ["zscore", "range" "auto"])} = "zscore"
                options.DataVariables (1,:) string ...
                    {mustBeInTable(tbl,options.DataVariables)}= ""
            end %arguments
            
            
            if options.DataVariables == ""
                options.DataVariables = util.isnumeric( tbl, ...
                    "asType", "string");
            else
                %Check if recognized name
            end
            
            if util.isvar( tbl, "Partition" )
                tFi = tbl.Partition == "Train";
            else
                tFi = true(height(tbl), 1);
            end
            
            n = numel( options.DataVariables );
            info =  table(  strings(n,1), nan(n,2), ...
                'VariableNames', ["Mthd", "Value"], ...
                'RowNames', options.DataVariables);
            
            switch mthd
                case "auto"
                    
                    tF = util.isnormal( tbl, ...
                        "DataVariables", options.DataVariables );
                    
                    %If normal, zscore
                    this = options.DataVariables(tF);
                    for iVar = this(:)'
                        
                        info{iVar,"Mthd"} = "zscore";
                        
                        [tbl.(iVar)(tFi), info{iVar,"Value"}] = ...
                            scalerhelper( tbl.(iVar)(tFi), "zscore" );
                        
                    end %for iVar
                    
                    %If not normal, range
                    this = options.DataVariables(~tF);
                    for iVar = this(:)'
                        
                        info{iVar,"Mthd"} = "range";
                        
                        [tbl.(iVar)(tFi), info{iVar,"Value"}] = ...
                            scalerhelper( tbl.(iVar)(tFi), "range" );
                        
                    end %for iVar
                    
                case {"zscore", "range"}
                    
                    for iVar = options.DataVariables(:)'
                        
                        info.Mthd = repelem(mthd, height(info), 1);
                        
                        [tbl.(iVar)(tFi), info{iVar,"Value"}] = ...
                            scalerhelper( tbl.(iVar)(tFi), mthd );
                        
                    end %for iVar
                    
                otherwise
                    error("Unhandled normalization method")
                    
            end %switch
            
            %Scale test partition
            if any( ~tFi )
                tbl(~tFi,:) = util.scaler(tbl(~tFi,:),info);
            end
            
            value = tbl;
            
        end %function
        
        
        function value = scaler(tbl, instructions)
            %SCALER Apply prior scaling to new data. A common use case is
            %to apply scaling used for training to new data in prediction.
            %
            % Syntax:
            %     value = util.scaler( tbl, instructions ) applies prior
            %     scaling instructions to data in tbl. Note these
            %     instructions are generated as the output of
            %     util.normalize (e.g. info output)
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                instructions table
            end %arguments
            
            vars = string( instructions.Properties.RowNames );
            for iVar = vars(:)'
                
                mthd  = instructions{iVar, "Mthd"};
                param = instructions{iVar, "Value"};
                
                switch mthd
                    case "zscore"
                        tbl.(iVar) = (tbl.(iVar) - param(1))./param(2);
                    case "range"
                        tbl.(iVar) = (tbl.(iVar) - param(1))./( param(2) - param(1) );
                    otherwise
                        error( "Unhandled scaler method." )
                end %switch
                
            end %for iVar
            
            value = tbl;
            
        end %function
        
        
        function value = skewness( tbl, options)
            %SKEWNESS Report skewness for numeric features
            %
            % Syntax:
            %   value = skewness( tbl )
            %   value = skewness( tbl, "DataVariables", varnames )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.DataVariables (1,:) string ...
                    {mustBeInTable(tbl,options.DataVariables)} = ""
            end %arguments
            
            if options.DataVariables == ""
                
                options.DataVariables = util.isnumeric( tbl, ...
                    "asType", "string");
                
            end %if options.DataVariables
            
            value = varfun(@skewness, tbl, "InputVariables", options.DataVariables);
            
            value.Properties.VariableNames = strrep( value.Properties.VariableNames, "skewness_", "" );
            
        end %function
        
        
        function [tbl, info] = log10( tbl, options )
            %LOG10 Base 10 logarithm applied to all numeric features, or a
            %specified subset.
            %
            % Syntax:
            %   [tbl, info] = log10( tbl ) apply log base10 transform to all
            %   numeric features in tbl
            %   [tbl, info] = log10( tbl, "DataVariables", varnames, "Jitter", true|false)
            %   apply transform to subset of features in DataVariables.
            %   Jitter specifies whether to replace zero values with 'eps-like'
            %   value to avoid -Inf condition.
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.DataVariables (1,:) string {mustBeInTable(tbl,options.DataVariables)} = ""
                options.Jitter (1,1) logical = true
            end %arguments
            
            if options.DataVariables == ""
                
                options.DataVariables = util.isnumeric( tbl, ...
                    "asType", "string");
                
            end %if options.DataVariables
            
            for iVar = options.DataVariables
                
                this = tbl.(iVar);
                
                %Negative value condition
                if any(this <0 )
                    this = 1+this-nanmin(this);
                end
                
                %Zero value condition
                if options.Jitter
                    this(this==0) = 1e-6;
                end %if options.Jitter
                
                tbl.(iVar) = log10(this);
                
            end %for iVar
            
            info = array2table( repelem( "log10", 1, numel(options.DataVariables) ), ...
                'VariableNames', options.DataVariables );
            
        end %function
        
        
        function [tbl, info] = pow( tbl, options )
            %POW Power transform appiled to all numeric features, or a
            %specified subset.
            %
            % Syntax:
            %   [tbl, info] = pow( tbl )
            %   [tbl, info] = pow( tbl, "DataVariables", varnames )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.DataVariables (1,:) string {mustBeInTable(tbl,options.DataVariables)} = ""
            end %arguments
            
            if options.DataVariables == ""
                
                options.DataVariables = util.isnumeric( tbl, ...
                    "asType", "string");
                
            end %if options.DataVariables
            
            for iVar = options.DataVariables
                
                this = tbl.(iVar);
                tbl.(iVar) = power(this,2);
                
            end %for iVar
            
            info = array2table( repelem( "pow", 1,numel(options.DataVariables) ), ...
                'VariableNames', options.DataVariables );
            
        end %function
        
        
        function [tbl, featurenames] = rmconstant( tbl, options )
            %RMCONSTANT Remove columns with constant entries
            %
            % Syntax:
            %   [value, featurenames] = util.rmconstant( value );
            %   [value, featurenames] = util.rmconstant( value, "DataVariables",... );
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.DataVariables (1,:) string {mustBeInTable(tbl,options.DataVariables)} = ""
            end
            
            if options.DataVariables == ""
                
                options.DataVariables = util.isnumeric( tbl, ...
                    "asType", "string");
                
            end %if options.DataVariables

            featurenames = options.DataVariables;
            
            %Remove constant vars
            tF1 = varfun(@(x)all(x == x(1)), tbl(:, featurenames), "OutputF", "uni");
            tbl(:, featurenames(tF1)) = [];
            featurenames = featurenames(~tF1);

            % Remove NaN features, if any
            tF2 = varfun(@(x) all(isnan(x)), tbl, ...
                "InputVariables", featurenames, ...
                "OutputFormat", "uniform");
            tbl(:, featurenames(tF2)) = [];
            featurenames = featurenames(~tF2);
                        
        end %function

        function [tbl, featurenames] = rminterclassconstant( tbl, options )
            %RMCONSTANT Remove columns with inter-class constant entries
            %
            % Syntax:
            %   [value, featurenames] = util.rminterclassconstant( value );
            %   [value, featurenames] = util.rminterclassconstant( value, "DataVariables",... );
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.DataVariables (1,:) string {mustBeInTable(tbl,options.DataVariables)} = ""
                options.ResponseVariable (1,1) string {mustBeInTable(tbl, options.ResponseVariable)} = ...
                    tbl.Properties.VariableNames(end)
            end
            
            if options.DataVariables == ""
                
                options.DataVariables = util.isnumeric( tbl, ...
                    "asType", "string");
                
            end %if options.DataVariables
            
            %Remove vars that are constant within a specific class
            G = findgroups(tbl.(options.ResponseVariable));

            constCheck = varfun(@(x)splitapply(@(x)all(x == x(1)), x, G), tbl(:, options.DataVariables), "OutputFormat", "table");
            tF = any(constCheck.Variables);

            tbl(:, options.DataVariables(tF)) = [];
            
            %Update feature names
            featurenames = options.DataVariables(~tF);
            
        end %function
        
        
        function result = sequence2frame( tbl, nWindow, nOverlap, options )
            %SEQUENCE2FRAME Buffer variables into data frames
            %
            % Syntax:
            %   value = util.sequence2frame(tbl, nWindow, nOverlap)
            %   value = util.sequence2frame(tbl, nWindow, nOverlap, ...
            %       "InputVariables", varnames, "GroupingVariables", varnames, ...
            %       "Impute", [true|false])
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                nWindow double {mustBeNumeric, mustBePositive, mustBeReal}
                nOverlap double {mustBeNumeric, mustBeReal} = 0
                options.InputVariables (1,:) string {validateattributes(options.InputVariables, ...
                    ["cellstr", "char", "string"], "nonempty"),...
                    mustBeInTable(tbl,options.InputVariables)} = ""
                options.GroupingVariables (1,:) string {validateattributes(options.GroupingVariables, ...
                    ["cellstr", "char", "string"], "nonempty"),...
                    mustBeInTable(tbl,options.GroupingVariables)} = ""
                options.Impute (1,1) logical = true
            end
            
            %Variables to Keep [Implicit | Explicit] Case
            if options.InputVariables == ""
                tF = varfun(@(x)isnumeric(x), tbl, 'OutputFormat','uniform');
            else
                tF = ismember( tbl.Properties.VariableNames, options.InputVariables );
            end
            
            %Create buffer
            thisFcn = @(x)bufferhelper(x, nWindow, nOverlap, options.Impute);
            
            if options.GroupingVariables == ""
                result  = varfun(thisFcn, tbl(:,tF), 'OutputFormat', 'table');
                
                result.Properties.VariableNames = tbl.Properties.VariableNames(tF);
            else
                [groups, meta] = findgroups( tbl(:, options.GroupingVariables) );
                uid = unique( groups );
                
                counter = 0;
                sequences = cell( numel( uid ), 1 );
                for igroup = uid(:)'
                    counter = counter + 1;
                    features = varfun(thisFcn, tbl(groups == igroup,tF), 'OutputFormat', 'table');
                    
                    features.Properties.VariableNames = tbl.Properties.VariableNames(tF);
                    
                    counter = counter + 1;
                    
                    metadata =  repmat( meta(igroup,:), height(features), 1  );
                    
                    % Check for overlapping variable names when appending
                    % metadata
                    overlapidx = ismember(metadata.Properties.VariableNames, features.Properties.VariableNames);
                    if any(overlapidx)
                        metadata.Properties.VariableNames(overlapidx) = ...
                            metadata.Properties.VariableNames(overlapidx) + "_1";
                    end
                    
                    sequences{ counter } = horzcat( ...
                        metadata, ...
                        features);
                end
                
                result = vertcat( sequences{:} );
            end
            
        end %function
        
        
        function result = sequence2ensemble( tbl, options )
            %SEQUENCE2ENSEMBLE Convert table to ensemble form
            %
            % Syntax:
            %   value = util.sequence2ensemble( tbl )
            %   value = util.sequence2ensemble( tbl, "GroupingVariable", varnames, ...
            %       "ExpandVariables", false|true )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.GroupingVariable(1,:) string {validateattributes(options.GroupingVariable, ...
                    ["cellstr", "char", "string"], "nonempty"), ...
                    mustBeInTable(tbl,options.GroupingVariable)} = ""
                options.ExpandVariables(1,1) logical = false
            end
            
            result = table();
            
            if options.GroupingVariable == ""
                [G, ID] = findgroups( tbl{:,end} );
            else
                [G, ID] = findgroups( tbl(:,options.GroupingVariable) );
            end
            
            varNames = tbl.Properties.VariableNames;
            tF = contains(varNames, options.GroupingVariable);
            
            if ~istimetable( tbl )
                timechk = any(contains(varNames,"Time",'IgnoreCase',true));
            else
                timechk = true;
            end
            
            if ~options.ExpandVariables
                result.Data = arrayfun(@(x) tbl(G == x, ~tF), unique(G), 'UniformOutput',false);
            else
                fun = @(x) ensemblehelper(x, tbl, varNames(~tF), timechk, G);
                result = arrayfun(fun,unique(G), ...
                    'UniformOutput',false);
                result = vertcat(result{:});
                
            end
            
            result = [result, ID];
            
        end %function
        
        
        function value = ensemble2sequence( ens, options )
            %ENSEMBLE2TABLE Convert ensemble to table form
            %
            % Syntax:
            %   value = ensemble2sequence( ens )
            %   value = ensemble2sequence( ens, "ConstantVariables", varnames)
            %
            
            arguments
                ens {mustBeClass(ens, ["table" "timetable"])}
                options.ConstantVariables (1,:) string {mustBeInTable(ens,options.ConstantVariables)} = ""
            end
            
            value = vertcat( ens.Data{:} );
            
            if options.ConstantVariables ~= ""
                n = rowfun(@height, ens, "InputVariables", "Data", ...
                    "ExtractCellContents", true, "OutputFormat", "uniform");
            end
            
            for iVariable = options.ConstantVariables(:)'
                value.(iVariable) = repelem( ens.(iVariable), n, 1 );
            end
            
            
        end %function
        
        
        function tbl = text2categorical( tbl, options )
            %TEXT2CATEGORICAL
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.DataVariables (1,:) string {mustBeInTable(tbl,options.DataVariables)} = ""
            end
            
            if options.DataVariables == ""
                
                options.DataVariables = util.name( tbl );
                
            end %if options.DataVariables
               
            tF = util.istext( tbl(:,options.DataVariables) );
               
            for iVar = options.DataVariables(tF)
                
                if ischar( tbl.(iVar) )
                    this = unique( cellstr(tbl.(iVar )) ); 
                    tbl.(iVar) = categorical( cellstr(tbl.(iVar)), this );
                else
                     this = unique( tbl.(iVar ) ); 
                    tbl.(iVar) = categorical( tbl.(iVar), this );
                end
                     
            end %for 
                
        end %function
        
        
        function [value, featurenames] = fscchi2( tbl, options )
            % FSCCHI2 Univariate "feature-based" selection method
            % (versus model-based) for classification using Chi-2 test
            %
            % Syntax:
            %   [value, featurenames] = util.fscchi2( tbl )
            %   [value, featurenames] = util.fscchi2( __, ...
            %       "PredictorNames", varnames, "ResponseNames", varname )
            %   [value, featurenames] = util.fscchi2( __, ...
            %       "ToKeep", percentToKeep )
            %   [value, featurenames] = util.fscchi2( __, ...
            %       "Plot", false|true )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.PredictorNames (1,:) string {mustBeInTable(tbl,options.PredictorNames)} = ...
                    tbl.Properties.VariableNames(1:end-1)
                options.ResponseName (1,:) string {mustBeInTable(tbl,options.ResponseName)} = ...
                    tbl.Properties.VariableNames( end )
                options.ToKeep (1,1) {mustBeInRange(options.ToKeep,0,100)} = 50
                options.Plot (1,1) logical = false
            end %arguments
            
            %Map
            features = tbl(:, options.PredictorNames);
            response = tbl.(options.ResponseName);
            
            %Confirm classification
            if isnumeric( response )
                error( "Feature selection method: fscchi2 is only valid for classification." )
            end %if isnumeric( response )
            
            %Attributes
            vars = util.name( tbl );
            attributenames = vars(...
                ~ismember(vars, [options.PredictorNames options.ResponseName]) );
            
            %Univariate feature selection
            [idx, scores] = fscchi2( features, response );
            
            %Preview feature ranking, including inf
            if options.Plot == true
                
                figure( "Color", "W" )
                bar( scores(idx) )
                xlabel( "Feature rank" )
                ylabel( "Feature importance" )
                xticks( 1:numel(idx) )
                xticklabels( strrep(features.Properties.VariableNames(idx), '_', ' ') )
                xtickangle(45)
                
                idxInf = find( isinf(scores) );
                
                if ~isempty( idxInf )
                    
                    hold on
                    if length(idxInf) == length(idx)
                        bar(1000*ones(length(idxInf),1)) %default Inf to 1000
                    else
                        bar(scores(idx(length(idxInf)+1))*ones(length(idxInf),1))
                    end
                    legend('Finite Scores','Inf Scores')
                    hold off
                    
                end
            end %if options.Plot == true
            
            %Drop features below specified threshold
            thresh = prctile(scores, options.ToKeep);
            tokeep = idx( scores(idx)>= thresh );
            
            %Final set of features
            featuresToKeep = features(:,tokeep);
            
            %Reconstruct data
            if ~isempty( attributenames )
                
                tF = contains(attributenames, "Partition");
                
                value = [...
                    tbl(:, attributenames(~tF) )...
                    featuresToKeep ...
                    tbl(:, attributenames(tF) ) ...
                    tbl(:,options.ResponseName) ];
            else
                value = [...
                    featuresToKeep ...
                    tbl(:,options.ResponseName)];
            end  %if options.AttributeNames
            
            %Update featurenames
            featurenames = string( featuresToKeep.Properties.VariableNames );
            
        end %function
        
        
        function [value, featurenames] = fscmrmr( tbl, options )
            % FSCMRMR Rank features for classification using minimum
            % redundancy maximum relevance (MRMR) algorithm
            %
            % Syntax:
            %   [value, featurenames] = util.fscmrmr( tbl )
            %   [value, featurenames] = util.fscmrmr( __, ...
            %       "PredictorNames", varnames, "ResponseNames", varname )
            %   [value, featurenames] = util.fscmrmr( __, ...
            %       "ToKeep", percentToKeep )
            %   [value, featurenames] = util.fscmrmr( __, ...
            %       "Plot", false|true )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.PredictorNames (1,:) string {mustBeInTable(tbl,options.PredictorNames)} = ...
                    tbl.Properties.VariableNames(1:end-1)
                options.ResponseName (1,:) string {mustBeInTable(tbl,options.ResponseName)} = ...
                    tbl.Properties.VariableNames( end )
                options.ToKeep (1,1) {mustBeInRange(options.ToKeep,0,100)} = 50
                options.Plot (1,1) logical = false
            end %arguments
            
            %Map
            features = tbl(:, options.PredictorNames);
            response = tbl.(options.ResponseName);
            
            %Confirm classification
            if isnumeric( response )
                error( "Feature selection method: fscmrmr is only valid for classification." )
            end %if isnumeric( response )
            
            %Attributes
            vars = util.name( tbl );
            attributenames = vars(...
                ~ismember(vars, [options.PredictorNames options.ResponseName]) );
            
            %Univariate feature selection
            [idx, scores] = fscmrmr( features, response );
            
            %Preview feature ranking, including inf
            if options.Plot == true
                
                figure( "Color", "W" )
                bar( scores(idx) )
                xlabel( "Feature rank" )
                ylabel( "Feature importance" )
                xticks( 1:numel(idx) )
                xticklabels( strrep(features.Properties.VariableNames(idx), '_', ' ') )
                xtickangle(45)
                
                idxInf = find( isinf(scores) );
                
                if ~isempty( idxInf )
                    
                    hold on
                    if length(idxInf) == length(idx)
                        bar(1000*ones(length(idxInf),1)) %default Inf to 1000
                    else
                        bar(scores(idx(length(idxInf)+1))*ones(length(idxInf),1))
                    end
                    legend('Finite Scores','Inf Scores')
                    hold off
                    
                end
            end %if options.Plot == true
            
            %Drop features below specified threshold
            thresh = prctile(scores, options.ToKeep);
            tokeep = idx( scores(idx)>= thresh );
            
            %Final set of features
            featuresToKeep = features(:,tokeep);
            
            %Reconstruct data
            if ~isempty( attributenames )
                
                tF = contains(attributenames, "Partition");
                
                value = [...
                    tbl(:, attributenames(~tF) )...
                    featuresToKeep ...
                    tbl(:, attributenames(tF) ) ...
                    tbl(:,options.ResponseName) ];
            else
                value = [...
                    featuresToKeep ...
                    tbl(:,options.ResponseName)];
            end  %if options.AttributeNames
            
            %Update featurenames
            featurenames = string( featuresToKeep.Properties.VariableNames );
            
        end %function
        
        
        function [value, featurenames] = fscanova( tbl, options )
            % FSCANOVA Compute ranking score using One-Way ANOVA.
            %
            % Syntax:
            %   [value, featurenames] = util.anova( tbl )
            %   [value, featurenames] = util.anova( __, ...
            %       "PredictorNames", varnames, "ResponseNames", varname )
            %   [value, featurenames] = util.anova( __, ...
            %       "ToKeep", percentToKeep )
            %   [value, featurenames] = util.anova( __, ...
            %       "Plot", false|true )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.PredictorNames (1,:) string {mustBeInTable(tbl,options.PredictorNames)} = ...
                    tbl.Properties.VariableNames(1:end-1)
                options.ResponseName (1,:) string {mustBeInTable(tbl,options.ResponseName)} = ...
                    tbl.Properties.VariableNames( end )
                options.ToKeep (1,1) {mustBeInRange(options.ToKeep,0,100)} = 50
                options.Plot (1,1) logical = false
            end %arguments
            
            %Map
            features = tbl(:, options.PredictorNames);
            response = tbl.(options.ResponseName);
            
            %Confirm classification
            if isnumeric( response )
                error( "Feature selection method: fscanova is only valid for classification." )
            end %if isnumeric( response )
            
            %Attributes
            vars = util.name( tbl );
            attributenames = vars(...
                ~ismember(vars, [options.PredictorNames options.ResponseName]) );
            
            % Convert label to numeric values
            if iscategorical(response)
                label = string(response);
            end
            group = grp2idx(label);
            
            %= One-way ANOVA
            % Normalize features using minmax.
            selectedFeatureValues = features.Variables;
            fNorm = (selectedFeatureValues-min(selectedFeatureValues,[],1))./(max(selectedFeatureValues,[],1)-min(selectedFeatureValues,[],1));
            
            % Compute ranking score using One-Way ANOVA.
            numFeatures = size(fNorm,2);
            scores = zeros(numFeatures,1);
            for k = 1:numFeatures
                [~,anovatbl] = anova1(fNorm(:,k),group,'off');
                % Get the F-statistic from the output of one-way ANOVA.
                stats = anovatbl{2,5};
                if ~isempty(stats)
                    scores(k) = stats;
                end
            end
            
            [~, idx] = sort(scores, 'descend');
            
            %Preview feature ranking, including inf
            if options.Plot == true
                
                figure( "Color", "W" )
                bar( scores(idx) )
                xlabel( "Feature rank" )
                ylabel( "Feature importance" )
                xticks( 1:numel(idx) )
                xticklabels( strrep(features.Properties.VariableNames(idx), '_', ' ') )
                xtickangle(45)
                
                idxInf = find( isinf(scores) );
                
                if ~isempty( idxInf )
                    
                    hold on
                    if length(idxInf) == length(idx)
                        bar(1000*ones(length(idxInf),1)) %default Inf to 1000
                    else
                        bar(scores(idx(length(idxInf)+1))*ones(length(idxInf),1))
                    end
                    legend('Finite Scores','Inf Scores')
                    hold off
                    
                end
            end %if options.Plot == true
            
            %Drop features below specified threshold
            thresh = prctile(scores, options.ToKeep);
            tokeep = idx( scores(idx)>= thresh );
            
            %Final set of features
            featuresToKeep = features(:,tokeep);
            
            %Reconstruct data
            if ~isempty( attributenames )
                
                tF = contains(attributenames, "Partition");
                
                value = [...
                    tbl(:, attributenames(~tF) )...
                    featuresToKeep ...
                    tbl(:, attributenames(tF) ) ...
                    tbl(:,options.ResponseName) ];
            else
                value = [...
                    featuresToKeep ...
                    tbl(:,options.ResponseName)];
            end  %if options.AttributeNames
            
            %Update featurenames
            featurenames = string( featuresToKeep.Properties.VariableNames );
            
        end %function
        
        
        function [value, featurenames, dr] = pca(tbl, options)
            %PCA Principal component dimension reduction
            %
            % Syntax:
            %   [value, featurenames, dr] = util.pca( tbl )
            %   [value, featurenames, dr] = util.pca( __, ...
            %       "PredictorNames", varnames, ...
            %       "Normalize", ["off"|"zscore"|"range"...])
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.PredictorNames (1,:) string {mustBeInTable(tbl,options.PredictorNames)} = ""
                options.Normalize(1,1) string ...
                    {mustBeMember(options.Normalize,["off", "zscore", "norm", "center", "scale", "range"])} = "off";
                options.ShowPareto (1,1) logical = false
            end %arguments
            
            if options.PredictorNames == ""
                featurenames = util.isnumeric( tbl, ...
                    "asType", "string");
            else
                featurenames = util.isnumeric( tbl(:, options.PredictorNames), ...
                    "asType", "string");
            end
            
            othernames = setdiff(util.name(tbl), ...
                featurenames, 'stable');
            
            %Remove any rows with nan in numeric feature set.
            tbl = rmmissing( tbl, "DataVariables", featurenames);
            
            features = tbl(:, featurenames);
            
            if width(features) > 10
                nComponents = 10;
            else
                nComponents = width(features);
            end %if width(features)
            
            dr = extract.pca( features,...
                'Normalize', options.Normalize, ...
                'NumComponents', nComponents );
            
            components = dr.table();
            componentfeaturenames = string(components.Properties.VariableNames);
            
            if options.ShowPareto
                fH = figure("Color", "W");
                p = pareto(dr.explained, 1);
                ax = p.Parent;
                
                set(p(2),...
                    'LineWidth', 1,...
                    'Color', [0 0.45 0.74],...
                    'Marker', 'o', ...
                    'MarkerSize', 6, ...
                    'MarkerFaceColor', [0 0.45 0.74], ...
                    'MarkerEdgeColor',[0 0.45 0.74]);
                
                set(p(1),...
                    'FaceColor', [0 0.45 0.74], ...
                    'FaceAlpha', .8);
                
                A = findall(fH, "Type", "Axes", 'YAxisLocation', 'right');
                set(A, 'visible', 'off')
                
                set(ax, 'YTick', (0:10:100))
                set(ax, 'YTickLabel', (0:10:100) + "%")
                
                ax.XTickLabel = "Component " + (1:length(ax.XTick));
                ax.XTickLabelRotation = 45;
                
                title(ax, "PCA Variance Explained")
                
            end %if
            
            value = horzcat( ...
                tbl(:, othernames), ...
                components);
            
            featurenames = componentfeaturenames;
            
        end %function
        
        
        function [value, featurenames] = cmds(tbl, mthd, options)
            %CMDS Classic multidimensional scaling dimension reduction
            %
            % Syntax:
            %   [value, featurenames] = util.cmds( tbl )
            %   [value, featurenames] = util.cmds( tbl, mthd )
            %   [value, featurenames] = util.cmds( __, ...
            %       "PredictorNames", varnames)
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                mthd (1,1) string {mustBeMember(mthd, ["euclidean", "squaredeuclidean", "seuclidean", ...
                    "cityblock", "minkowski", "chebychev"])} = "euclidean"
                options.PredictorNames (1,:) string {mustBeInTable(tbl,options.PredictorNames)} = ""
            end %arguments
            
            if options.PredictorNames == ""
                featurenames = util.isnumeric( tbl, ...
                    "asType", "string");
            else
                featurenames = util.isnumeric( tbl(:, options.PredictorNames), ...
                    "asType", "string");
            end
            
            othernames = setdiff(util.name(tbl), ...
                featurenames, 'stable');
            
            %Remove any rows with nan in numeric feature set.
            tbl = rmmissing( tbl, "DataVariables", featurenames);
            
            features = tbl(:, featurenames);
            
            %Dimension Reduction
            distances   = pdist( features.Variables, mthd );
            [result, ~] = cmdscale( distances );
            
            mds = array2table(result, 'VariableNames', ("Dimension "+(1:size(result,2))) );
            mdsfeaturenames = string(mds.Properties.VariableNames);
            
            value = horzcat( ...
                tbl(:, othernames), ...
                mds);
            
            featurenames = mdsfeaturenames;
            
        end %function
        
        
        function [value, featurenames] = descriptivestatistics( tbl, options )
            %DESCRIPTIVESTATISTICS Calculate descriptive statistics on buffer/frames
            %
            % Syntax:
            %   [value, featurenames] = util.descriptivestatistics( tbl )
            %   [value, featurenames] = util.descriptivestatistics( __, ...
            %       "InputVariables", varnames)
            %   [value, featurenames] = util.descriptivestatistics( __, ...
            %       "Statistics", statisticnames )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.InputVariables (1,:) string {mustBeInTable(tbl,options.InputVariables)} = ""
                options.Statistics (1,:) string {mustBeMember(options.Statistics, ...
                    ["all"
                    "mean"
                    "range"
                    "iqr"
                    "std"
                    "min"
                    "max"
                    "mad"
                    "skewness"
                    "kurtosis"
                    "p10"
                    "median"
                    "p90"])} = "all"
            end
            
            %Specify features
            if options.InputVariables == ""
                tF = util.isnumeric( tbl,  "asType", "logical");
            else
                tF = ismember( tbl.Properties.VariableNames, options.InputVariables );
            end %if options.DataVariables
            
            featurenames = string( tbl.Properties.VariableNames( tF ) );
            
            %Function Array
            nameArray(1)    = "mean";
            nameArray(2)    = "range";
            nameArray(3)    = "iqr";
            nameArray(4)    = "std";
            nameArray(5)    = "min";
            nameArray(6)    = "max";
            nameArray(7)    = "mad";
            nameArray(8)    = "skewness";
            nameArray(9)    = "kurtosis";
            nameArray(10)   = "p10";
            nameArray(11)   = "median";
            nameArray(12)   = "p90";
            
            fcnArray{1}     = @(x)mean(x, 2 );
            fcnArray{2}     = @(x)range(x, 2);
            fcnArray{3}     = @(x)iqr(x, 2);
            fcnArray{4}     = @(x)std(x, [], 2);
            fcnArray{5}     = @(x)min(x, [], 2);
            fcnArray{6}     = @(x)max(x, [], 2);
            fcnArray{7}     = @(x)mad(x, 0, 2);
            fcnArray{8}     = @(x)skewness(x, 1, 2);
            fcnArray{9}     = @(x)kurtosis(x, 1, 2);
            fcnArray{10}    = @(x)prctile(x, 10, 2);
            fcnArray{11}    = @(x)median(x, 2);
            fcnArray{12}    = @(x)prctile(x, 90, 2);
            
            %Select a subset of statistics
            if options.Statistics ~= "all"
                
                tF_statistic = ismember( nameArray, options.Statistics );
                fcnArray     = fcnArray( tF_statistic );
                nameArray    = nameArray( tF_statistic );
                
            end %if options.Statistics
            
            value = tbl;
            
            for iFcn = 1 : numel( fcnArray )
                
                thisResult  = varfun(fcnArray{iFcn}, tbl(:, featurenames), 'OutputFormat', 'table');
                thisName    = thisResult.Properties.VariableNames;
                
                thisResult.Properties.VariableNames = strrep(thisName,'Fun',nameArray(iFcn));
                
                value = [value thisResult]; %#ok<AGROW>
                
            end %for iFcn
            
            %Update feature names
            vars = string( value.Properties.VariableNames );
            tF = contains(vars, "_" + featurenames );
            featurenames =  vars(tF);
            
        end %function
        
        
        function value = balance( tbl, options )
            %BALANCE Undersample imbalanced data labels
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.Response (1,1) string {mustBeInTable(tbl,options.Response)}
                options.Seed (1,1) double = 0
            end

            stat = groupcounts(tbl, options.Response);              
            n    = min( stat.GroupCount ) ;

            classes = stat.( options.Response );
            
            iD = [];
            for iClass = classes(:)'
                
                where = find( tbl.( options.Response ) == iClass ); 
 
                rng( options.Seed )
                index = datasample(where, n, "Replace", false);
                
                iD = [iD; index];  %#ok<AGROW>
                
            end
            
            value = tbl(iD,:);
            
        end %function 
        
        
        function value = isnormal( tbl, options )
            %ISNORMAL True for features with normal distribution
            %
            % Syntax:
            %   value = util.isnormal( tbl )
            %   value = util.isnormal( tbl, "DataVariables", varnames, ...
            %       "asType", "logical" | "string" )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.DataVariables (1,:) string {mustBeInTable(tbl,options.DataVariables)} = ""
                options.asType (1,1) ...
                    {mustBeMember(options.asType,["logical", "string"])} = "logical"
            end
            
            if options.DataVariables == ""
                
                options.DataVariables = util.isnumeric( tbl, ...
                    "asType", "string");
                
            end %if options.DataVariables
            
            value = varfun(@(x)~kstest(x), tbl, ...
                "InputVariables", options.DataVariables, ...
                "OutputFormat", "uniform");
            
            if options.asType == "string"
                value = string(tbl.Properties.VariableNames(value));
            end %if options.asType
            
        end %function
        
        
        function value = isnumeric(tbl, options)
            %ISNUMERIC True for numeric features
            %
            % Syntax:
            %   value = util.isnumeric( tbl )
            %   value = util.isnumeric( tbl, "asType", "logical" | "string" )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.asType (1,1) string...
                    {mustBeMember(options.asType,["logical", "string"])} = "logical"
            end %arguments
            
            value = varfun(@isnumeric, tbl, "OutputFormat", "uniform");
            
            if options.asType == "string"
                value = string(tbl.Properties.VariableNames(value));
            end %if options.asType
            
        end %function
        
        
        function value = iscategorical(tbl, options)
            %ISCATEGORICAL True for numeric features
            %
            % Syntax:
            %   value = util.iscategorical( tbl )
            %   value = util.iscategorical( tbl, "asType", "logical" | "string" )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.asType (1,1) string...
                    {mustBeMember(options.asType,["logical", "string"])} = "logical"
            end %arguments
            
            value = varfun(@iscategorical, tbl, "OutputFormat", "uniform");
            
            if options.asType == "string"
                value = string(tbl.Properties.VariableNames(value));
            end %if options.asType
            
        end %function
        
        
        function value = istext(tbl, options)
            %ISTEXT True if text based feature(e.g. not a categorical).
            %Includes strings, chars, cellstrs).
            %
            % Syntax:
            %    value = util.istext( tbl )
            %    value = util.istext( tbl, "asType", "logical" | "string" )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.asType (1,1) string...
                    {mustBeMember(options.asType,["logical", "string"])} = "logical"
            end %arguments
            
            value = varfun(@(x)iscellstr(x) | ischar(x) | isstring(x), tbl, ...
                "OutputFormat", "uni");
            
            if options.asType == "string"
                value = string(tbl.Properties.VariableNames(value));
            end %if options.asType
            
        end %function
        
        
        function tF = isconstant( tbl, options )
            %ISCONSTANT True for constant features
            %
            % Syntax:
            %   tF = util.isconstant( tbl )
            %   tF = util.isconstant( tbl, "Response", varname )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.Response (1,1) {mustBeClass(options.Response, "string"), ...
                    mustBeInTable(tbl,options.Response)} = ""
            end
            
            tF = varfun(@iscellstr, tbl, "OutputFormat", "uniform");
            
            for iVar = find(tF)
                tbl.(iVar) = string( tbl.(iVar) );
            end
            
            if options.Response == ""
                tF = varfun(@(x) all(all(x==x(1,:),2)), tbl, "OutputFormat", "uniform");
            else
                tF = varfun(@(x) all(all(x==x(1,:),2)), tbl, "OutputFormat", "uniform", ...
                    "GroupingVariables", options.Response);
            end
            
        end %function
        
        
        function tF = isvar( tbl, vars)
            %ISVAR True if specified names are in table
            %
            % Syntax:
            %   tF = util.isvar( tbl, vars )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                vars (1,:) string
            end
            
            tF = nnz( ismember(tbl.Properties.VariableNames, vars ) ) == numel(vars);
            
        end %function
        
        
        function value = name(tbl, options)
            %NAME Return table variable names
            %
            % Syntax:
            %   value = name( tbl )
            %   value = name( tbl, tF )
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                options.tF (1,:) logical = true(width(tbl),1);
            end
            
            if numel(options.tF) ~= width(tbl)
                error("UTIL:NAME:wrongsize", "numel(tF) must equal width(tbl)")
            end
            
            value = string(tbl.Properties.VariableNames(options.tF));
            
        end %function
        
        
        function value = custom(tbl, property)
            %custom Return custom table properties
            %
            % Syntax:
            %   value = util.custom( tbl ) returns custom properties assigned in
            %   table
            %   value = util.custom( tbl, property ) gets a specified property
            %
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                property (1,1) {mustBeClass(property,"string")} = ""
            end
            
            if property == ""
                value = tbl.Properties.CustomProperties;
            else
                this = string( properties( tbl.Properties.CustomProperties ) );
                tF = ismember(this, property);
                
                if any( tF )
                    value = tbl.Properties.CustomProperties.( this(tF) );
                end
                
            end
            
        end %function
        
    end %methods
    
    
end %classdef


%Helper functions
function [value, param] = scalerhelper( x, mthd )

switch mthd
    case "zscore"
        tF = ~isnan(x);
        value = nan(size(x));
        
        [value(tF), mu, sigma] = zscore( x(tF) );
        param = [mu sigma];
    case "range"
        mn = min(x, [], 'omitnan');
        mx = max(x, [], 'omitnan');
        
        value = (x - mn )./( mx - mn );
        param = [mn mx];
end %mthd

end %function


function [result, index] = bufferhelper( data, nWindow, nOverlap, isFill )

if isempty(nOverlap)
    %        indx = buffer( 1:numel(data), nWindow, 'nodelay' );
    indx = buffer( 1:numel(data), nWindow );
else
    %         indx = buffer( 1:numel(data), nWindow, nOverlap,'nodelay' );
    indx = buffer( 1:numel(data), nWindow, nOverlap );
end

tF = indx==0;
indx( tF ) = NaN;

if isFill
    indx  = fillmissing( indx, 'nearest', 1 );
    value = data(indx);
else
    value = nan( size(indx) );
    value( ~tF ) = data( indx(~tF) );
end

result = transpose(value);
index  = transpose(indx);

end %function


function dataRow = ensemblehelper(groups, data, vars, timechk, G)

data_temp = data(G == groups, vars);

if ~timechk
    dataRow = varfun(@(x) {x}, data_temp, ...
        'OutputFormat','table');
else
    if istimetable(data_temp)
        time = data_temp.Properties.RowTimes;
        dataRow = varfun(@(x) {timetable(time,x)},data_temp, ...
            'OutputFormat','table');
    else
        timeidx = contains(vars,"Time",'IgnoreCase',true);
        time = data_temp{:,timeidx};
        
        if ~(isdatetime(time) || isduration(time))
            error(['Time vector is not duration or datetime. ', ...
                'Please convert time vector to duration or datetime prior to running sequence2ensemble'])
        end
        
        dataRow = varfun(@(x) {timetable(time,x)},data_temp, ...
            'OutputFormat','table');
        dataRow = dataRow(:,2:end);
        
    end
    
end

dataRow.Properties.VariableNames = ...
    strrep(string(dataRow.Properties.VariableNames),"Fun_","");

end %function


%Validation functions
function mustBeClass( value, members )

if ~ismember(class( value ), members)
    
    msg = repelem( "%s",1,numel(members) );
    if numel(members) > 1
        msg = join(msg(1:end-1), ",") + ", or " + msg(end);
    end
    
    throwAsCaller( MException("util:mustBeClass", ...
        sprintf( 'Value must be of the following types: '+msg, members) ))
end

end %function


function mustBeInTable( tbl, value )

%Condition 1: unspecified
if value == ""
    return
end

%Condition 2: must be in table
tF = util.isvar( tbl, value );

if ~tF
    throwAsCaller( MException("util:mustBeInTable", ...
        "Specified variables not found in table.") )
end

end %function
