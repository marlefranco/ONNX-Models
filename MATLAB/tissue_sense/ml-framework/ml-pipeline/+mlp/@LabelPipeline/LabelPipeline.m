classdef LabelPipeline < dap.Pipeline 
    %LABELPIPELINE Unsupervised pipeline 
   
    properties
        Label mlp.Label
    end %public 


    methods
        function obj = LabelPipeline( varargin )
            %LABELPIPELINE Construct an instance of this class
            
            obj = obj@dap.Pipeline( varargin{:} );
                        
        end %Pipeline 
    end %constructor 
    
    
    %% Update
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
                catch
                    value  = obj.apply( functions, varargin{:} );
                end
            else   
                error( "Unsupported function signature. Please specify 1 or 2 outputs.") 
            end
              
            if istable( value )

                obj.Data = value;
                                    
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
                 
            elseif isvector( value ) && numel(value) == height(obj.Data)
                      
               checkpointSteps = obj.Step;
               if ~isempty(checkpointSteps)
                    theseSteps = checkpointSteps( [checkpointSteps.Type] == "Data" );
               else
                   theseSteps = checkpointSteps;
               end
               
               if exist( 'info', 'var' ) && istable(info)  
                   thisLabel = mlp.Label( 'label', value, 'pipe', theseSteps, 'metadata', info );
               else
                   thisLabel = mlp.Label( 'label', value, 'pipe', theseSteps);
               end
               obj.Label = [obj.Label thisLabel] ;  
                
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
            obj.Label = mlp.Label.empty;
            
        end %reset 
        

        function value = initialize( obj )
            
            value = {...
                "FeatureNames", obj.features};
        end
    

    end %public
    
    
    %% Public
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
        

        function result = describe( obj )
            %DESCRIBE
            
            result = [];
            for iObj = 1: numel(obj)
                
                thisReport = obj(iObj).Label.describe;
                result = [ result; thisReport  ]; %#ok<AGROW>
                
            end %iObj
            
        end %describe
        
        
        function result = lastfit( obj )
            %LASTMODEL 
            
            if ~isempty( obj.Label )
               result = obj.Label( end );
            else 
               result = [];  
            end
            
        end %lastfit
        
        function value = mdl( obj, index )     
            %MDL
            
            if isvalid( obj.Label(index) ) && ~isempty( obj.Label(index) )
                value =  obj.Label( index ).mdl;
            else
                value = [];
            end
        end %mdl
        
    end %public
    
    
    %% Static 
    methods (Static)
        
        function value = new( varargin )
            %NEW
            value = mlp.LabelPipeline( varargin{:} );
        end %new
        
        function value = restore( data ,steps, name )
            %RESTORE
            
            if nargin < 3
                name = "";
            end
            
            value = mlp.LabelPipeline( data, "Name", name );
            
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
            
            value = mlp.LabelPipeline.restore( data, steps, name );
            
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


