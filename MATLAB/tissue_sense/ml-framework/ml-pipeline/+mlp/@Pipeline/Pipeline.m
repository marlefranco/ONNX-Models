classdef Pipeline < dap.Pipeline
    %PIPELINE Summary of this class goes here
    %
    %
    % mlp.Pipeline methods:
    %   update -
    %   reset -
    %   initialize -
    %
    %   snapshot -
    %   custom -
    %   ignore -
    %   slice -
    %   original -
    %   dataset -
    %   training -
    %   testing -
    %   describe -
    %   lastmodel -
    %   mdl -
    %
    %   new -
    %   restore -
    %   load -
    %   listCustomProperties -
    
    properties
        Model
    end %public
    
    properties (Dependent)
        categoricalfeatures
    end
    
    properties (Dependent)
        response
        trainingobs
        testobs
    end
    
    methods
        function obj = Pipeline( varargin )
            %PIPELINE Construct an instance of this class
            obj = obj@dap.Pipeline( varargin{:} );
            
            %Attach Custom Properties
            obj.Data = addprop(obj.Data, "Evaluation", "table" );
            obj.Data = addprop(obj.Data, "Naive", "table" );
            obj.Data.Properties.CustomProperties.Evaluation = table();
            obj.Data.Properties.CustomProperties.Naive = [];
            
        end %Pipeline
    end %constructor
    
    methods
        function update( obj, functions, varargin )
            %UPDATE Run function in the pipeline
            
            obj.setPrior()
            
            nSteps = numel( obj.Step );
            
            nOutputs = nargout( functions );
            
            if nOutputs == 1
                value  = obj.apply( functions, varargin{:} );
            elseif nOutputs == 2
                [value, info]  = obj.apply( functions, varargin{:} );
            elseif nOutputs < 0
                try
                    [value, info]  = obj.apply( functions, varargin{:} );
                catch ME
                    if ME.message == "Too many output arguments."
                        value  = obj.apply( functions, varargin{:} );
                    else
                        throw( ME )
                    end
                end
            else
                error( "Unsupported function signature. Please specify 1 or 2 outputs.")
            end
            
            if istable( value )
                
                obj.Data = value;
                
                if exist( 'info' , 'var' )
                    if ~(isempty(obj.Model) && isempty(info))
                        obj.Model(end).testmetadata = info;
                    end
                end
                
                if numel( obj.Step ) == nSteps
                    
                    if contains( string( func2str(functions) ), "predict")
                        
                        newRecord = dap.stepInPipeline.Record( ...
                            'Operation', functions, ...
                            'Args', varargin, ...
                            'Type', "Prediction");
                        
                    else
                        
                        newRecord = dap.stepInPipeline.Record( ...
                            'Operation', functions, ...
                            'Args', varargin, ...
                            'Type', "Data");
                        
                    end
                    
                    obj.addStep( newRecord );
                end
                
            elseif isobject( value )
                
                checkpointSteps = obj.Step;
                theseSteps = checkpointSteps( [checkpointSteps.Type] == "Data" );
                
                if exist( 'info', 'var' )
                    thisModel = mlp.Model( 'mdl', value, 'pipe', theseSteps, 'metadata', info );
                else
                    thisModel = mlp.Model( 'mdl', value, 'pipe', theseSteps);
                end
                
                obj.Model = [obj.Model thisModel] ;
                
                if numel( obj.Step ) == nSteps
                    newRecord = dap.stepInPipeline.Record( ...
                        'Operation', functions, ...
                        'Type', "Model");
                    
                    obj.addStep( newRecord );
                end
                
            else
                error('Result is not a table or model')
            end
            
        end %update
        
        function reset(obj)
            
            reset@dap.Pipeline( obj )
            obj.Model = [];
            
        end %reset
        
        
        function value = initialize( obj, type )
            
            arguments
                obj
                type (1,1) string ...
                    {mustBeMember(type, ["standard" "pdm"])} = "standard"
            end %arguments
            
            switch type
                case "standard"
                    value = {...
                        "PredictorNames", obj.features, ...
                        "ResponseName", obj.response, ...
                        "Include", obj.trainingobs};
                case "pdm"
                    value = {...
                        "HealthIndicatorName", obj.Data.Properties.CustomProperties.HealthIndicatorName, ...
                        "DataVariable", obj.Data.Properties.CustomProperties.DataVariable, ...
                        "LifeTimeVariable", obj.Data.Properties.CustomProperties.LifeTimeVariable};
            end %switch
            
        end %function
        
    end %public
    
    methods
        function result = snapshot( obj )
            
            result = obj.Data;
            result=addprop(result, "History", "table" );
            result.Properties.CustomProperties.History = obj.history;
            
        end
        
        
        function result = custom( obj )
            %CUSTOM
            
            if ~isempty( obj.Data )
                result = obj.Data.Properties.CustomProperties;
            else
                result = [];
            end
            
        end %custom
        
        
        function result = ignore( obj, list )
            %IGNORE
            
            arguments
                obj
                list (1,:) {stringorhandleValidation( list ) } = obj.VariableNames
            end
            
            if isa( list, 'function_handle' )
                tF           = varfun(list, obj.Data, "OutputFormat", "uni");
                list         = obj.VariableNames(tF);
            end
            
            vars    = obj.VariableNames;
            index   = ~ismember( vars, list );
            result  = vars( index );
        end %ignore
        
        
        function result = slice( obj, options )
            %SLICE
            
            arguments
                obj
                options.vars (1,:){stringorhandleValidation( options.vars ) } = obj.vars
                options.rows (1,1) string = ""
            end
            
            if isa( options.vars, 'function_handle' )
                tF              = varfun(options.vars, obj.Data, "OutputFormat", "uni");
                options.vars    = obj.VariableNames(tF);
            end
            
            rowsToKeep = true( height( obj.Data ), 1 );
            if options.rows ~= ""
                eObj            = ExpressionEvaluator();
                eObj.SourceData = obj.Data;
                rowsToKeep      = eObj.eval( options.rows );
            end
            
            result = obj.Data( rowsToKeep, options.vars );
            
        end %slice
        
        function result = original( obj )
            
            result = obj.Source;
            
        end %original
        
        function result = dataset( obj, list )
            
            arguments
                obj
                list (1,:) {stringorhandleValidation( list ) } = [obj.features obj.response]
            end
            
            if isa( list, 'function_handle' )
                subSet = [obj.features obj.response];
                tF     = varfun(list, obj.Data(:,subSet), "OutputFormat", "uni");
                list   = subSet(tF);
            end
            
            result = obj.Data(:, list);
            
        end
        
        function result = training( obj, options )
            %TRAINING
            
            arguments
                obj
                options.partition (1,1) string = "Partition";
            end
            
            expr    = sprintf( "%s == ""Train""", options.partition );
            result  = obj.slice( "vars", [obj.features obj.response], "rows", expr );
            
        end %training
        
        function result = testing( obj, options )
            %TESTING
            
            arguments
                obj
                options.partition (1,1) string = "Partition";
            end
            
            expr    = sprintf( "%s == ""Test""", options.partition );
            result  = obj.slice( "vars", [obj.features obj.response], "rows", expr );
            
        end %training
        
        
        function result = describe( obj )
            %DESCRIBE
            
            result = [];
            for iObj = 1: numel(obj)
                
                thisReport = obj(iObj).Model.describe;
                result = [ result; thisReport  ]; %#ok<AGROW>
                
            end %iObj
            
        end %describe
        
        
        function result = lastmodel( obj )
            %LASTMODEL
            
            if ~isempty( obj.Model )
                result = obj.Model( end );
            else
                result = [];
            end
            
        end %lastmodel
        
        function value = mdl( obj, index )
            %MDL
            
            if isvalid( obj.Model(index) ) && ~isempty( obj.Model(index) )
                value =  obj.Model( index ).mdl;
            else
                value = [];
            end
        end %mdl
        
        
    end %public
    
    methods
        
        function value = get.response( obj )
            
            if ~isempty( obj.Data )
                
                if contains( "Response", properties(obj.Data.Properties.CustomProperties) )
                    value = obj.Data.Properties.CustomProperties.Response;
                else
                    
                    if contains( "Partition", obj.Data.Properties.VariableNames )
                        value = string( obj.Data.Properties.VariableNames( end-1 ) );
                        %Store Response, Features, and Partition
                        obj.Data = addprop(obj.Data, "Response", "table" );
                        
                        obj.Data.Properties.CustomProperties.Response = value;
                        
                    else
                        value = string( obj.Data.Properties.VariableNames( end ) );
                        obj.Data = addprop(obj.Data, "Response", "table" );
                        
                        obj.Data.Properties.CustomProperties.Response = value;
                    end
                end
                
            else
                value = "";
            end
            
        end %response
        
        function value = get.trainingobs( obj )
            
            if ~isempty( obj.Data )
                
                if contains("Partition", obj.Data.Properties.VariableNames)
                    value = obj.Data.Partition == "Train";
                else
                    value = true(height(obj.Data), 1);
                end
                
            else
                value = [];
            end
            
        end %trainingobs
        
        function value = get.testobs( obj )
            
            if ~isempty( obj.Data )
                
                if contains("Partition", obj.Data.Properties.VariableNames)
                    value = obj.Data.Partition == "Test";
                else
                    value = false(height(obj.Data), 1);
                end
                
            else
                value = [];
            end
            
        end
        
        
        function value = get.categoricalfeatures( obj )
            if obj.Features ~=""
                tF = varfun(@iscategorical, obj.Data, ...
                    'InputVariables', obj.features, ...
                    'OutputF', 'uni');
                value = obj.Features( tF );
            else
                value = [];
            end
        end %categoricalfeatures
        
    end %get
    
    methods ( Access = private )
        function tbl = updateCustomProperties( obj, tbl )
            
            newCustom = mlp.Pipeline.listCustomProperties( tbl );
            
            listDifferences = string(setxor( properties(obj.custom), ...
                properties(newCustom) ));
            
            listIntersection = string( intersect( properties(obj.custom), ...
                properties(newCustom) ));
            
            for iProp = listDifferences(:)'
                
                tbl = addprop( tbl, iProp );
                tbl.Properties.CustomProperties( iProp ) = ...
                    obj.custom.( iProp );
                
            end %for iProp
            
            %Check for updates
            for iProp = listIntersection(:)'
                
                if ~isequal( tbl.Properties.CustomProperties.( iProp ), obj.custom.( iProp ) ) && ~isempty( obj.custom.( iProp ) )
                    tbl.Properties.CustomProperties.( iProp ) = ...
                        vertcat(obj.custom.( iProp ), tbl.Properties.CustomProperties.( iProp ));
                    
                end
                
            end
            
        end %updateCustomProperties
    end %private
    
    methods (Static)
        
        function value = new( varargin )
            %NEW
            value = mlp.Pipeline( varargin{:} );
        end %new
        
        function value = restore( data ,steps, name )
            %RESTORE
            
            if nargin < 3
                name = "";
            end
            
            value = mlp.Pipeline( data, "Name", name );
            
            nSteps = numel( steps );
            for iStep = 1 : nSteps
                steps( iStep ).restore( value )
            end
            
        end %restore
        
        function value = load( data, location )
            %LOAD
            
            importedSteps = load( location );
            
            steps = importedSteps.pipelineSteps;
            name  = importedSteps.name;
            
            value = mlp.Pipeline.restore( data, steps, name );
            
        end %load
        
        function result = listCustomProperties( tbl )
            result = tbl.Properties.CustomProperties;
        end %listCustomProperties
        
    end %static
    
end %classdef


% Custom validator functions
function stringorhandleValidation(input)
    % Test for specific class
    if ~isstring( input ) && ~isa( input, 'function_handle' )
        error('Input must be a function handle or string.')
    end
end 

% function handleValidation(input)
%     % Test for specific class
%     if  ~isa( input, 'function_handle' )
%         error('Input must be a function handle or string.')
%     end
%     
%     if numel( input ) > 1
%         error('Input must be scalar.')
%     end
% end 
