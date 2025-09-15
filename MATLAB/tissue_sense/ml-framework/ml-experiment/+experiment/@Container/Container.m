classdef (Hidden) Container < matlab.mixin.SetGet
    % experiment.Container Create an experiment configuration 
    %
    %   Required Parameter:
    %
    %   PipelineFcn: Anonymous function referencing pipeline in the form:
    %       @(data, settings)somePipeline(data, settings{:})...
    %       @(settings)somePipeline( settings{:} ) or 
    %
    %   Optional Parameters:
    %
    %   Description: Experiment description 
    %       [string]
    %
    %   PipelineConfiguration: list of parameters to optimize/sweep in pipeline 
    %       [optimzeParameter]
    %
    %   AdditionalConfiguration: list of contstant parameters for pipeline
    %       [constantParameter]
    %
    %   Search: Search optimization as gridsearch or randomsearch 
    %       [string]
    %
    %   Evaluations: Number of random samples if Search = "randomsearch"
    %       [integer]
    %
    %   NumGridDivisions: Number of values in each dimesion is Search = "gridsearch"
    %       [integer]
    %
    % Example 
    %   
    %   session = experiment.Container(...
    %       "Description", "Experiment Trials", ...
    %       "PipelineFcn", @(settings)somePipeline( settings{:} ), ...
    %       "PipelineConfiguration", PipelineParameters,...
    %       "AdditionalConfiguration", AdditionalParameters, ...
    %       "Search", "gridsearch", ...
    %       "NumGridDivisions", 2);
    %
    %
    %   N.Howes
    %   MathWorks Consulting 2020
    %
    
    properties
        Data
        Description
        PipelineFcn (1,:) function_handle {mustBeClass(PipelineFcn,"function_handle")} = function_handle.empty(1,0);
        PipelineConfiguration (1,:) optimizeParameter 
        AdditionalConfiguration (1,:) constantParameter
        Search (1,1) string {mustBeMember(Search, ["gridsearch", "randomsearch"])} = "gridsearch"
        Evaluations (1,1) double {mustBeInteger} = 30;
        NumGridDivisions (1,1) double {mustBeInteger} = 10;
    end
    
    properties (SetAccess = private, GetAccess = public )
        trials 
    end
    
    properties (SetAccess = private, GetAccess = public )
       Samples
       Constants
    end
    
    properties ( Dependent )
        validTrialIndices
        nSamples
    end
    
    methods
        function obj = Container(varargin)
            %Container Summary of this method goes here
   
            if nargin > 0
               
                if ~isempty( varargin )
                    set(obj, varargin{:})
                end
            end
            
        end %experiment.Container
    end
    
    methods
        
        function value = select( obj, options )
            
            arguments
                obj experiment.Container
                options.Id (1,:) double {mustBeInteger} = 1;
            end
            
            value = [ obj.Trials.Record{ options.Id } ];
      
        end %select 
            
        
        function reset( obj )
            
            obj.Trials = table( 'Size', [obj.nSamples, 2], 'VariableTypes', ["cell" "logical"],...
                'VariableNames', ["Record", "Prepared"], ...
                'RowNames', "Trial" + (1:obj.nSamples) );
            
            [obj.Trials.Record(:)] = deal( {struct()} );
            
        end %reset 
            
        
        function run( obj, options )

            arguments
                obj experiment.Container
                options.Id (1,:) double {mustBeInteger} = 1:obj.nSamples; 
            end
            
            selectedTrials = options.Id;
            
            for iTrial = selectedTrials(:)'
                
                if ~isempty( obj.Samples )
                    settings = namedargs2cell( obj.Samples( iTrial ) );
                else
                    settings = {};
                end
                
                if ~isempty( obj.Constants )
                    constants = namedargs2cell(obj.Constants( iTrial) );
                    settings = [settings   constants]; %#ok<AGROW>
                end
                
                if nargin( obj.PipelineFcn ) == 1
                    result = obj.PipelineFcn( settings );
                else
                    result = obj.PipelineFcn( obj.Data, settings );
                end
                
                obj.Trials.Record{ iTrial } = result;
                obj.Trials.Prepared( iTrial ) = true;
            end
            
        end %prepare
        
        
        function validate( obj )
            %VALIDATE
            
            parameters = [ ...
                [obj.PipelineConfiguration.Name] ...
                [obj.AdditionalConfiguration.Name] ...
                ];
            
            if ~isempty( parameters )

                str  = func2str( obj.PipelineFcn );
                expr = "@*\)([a-zA-Z]{1,1}[a-zA-Z0-9]+)(*";

                filename = string(...
                    regexp(str, expr, "tokens")...
                    );

                validparameters =  validatePipelineParameters( filename );

                parameters = [ ...
                    [obj.PipelineConfiguration.Name] ...
                    [obj.AdditionalConfiguration.Name] ...
                    ];

                isnotvalid = ~ismember( parameters, validparameters ); 

                if  any( isnotvalid )
                     error( ['Unhandled/unknown parameters in configuration:', ...
                         sprintf( '\n''%s''', parameters(isnotvalid))] )
                end
                
            end
 
        end %validate
        
        
        function build( obj )
            %BUILD Summary of this method goes here
            
            parameters = obj.PipelineConfiguration(:)';
            
            if ~isempty( obj.PipelineConfiguration )
            
                results = cell( size( parameters ) );
                levels = cell( size( parameters ) );

                types = [parameters.Type];
                tF = ismember(types, ["optimizeContinuous" "optimizeInteger"]);

                switch obj.Search

                    case "gridsearch"

                        %Continuous
                        [parameters(tF).Method] = deal("grid");
                        [parameters(tF).Samples] = deal(obj.NumGridDivisions);

                        %Discrete
                        [parameters(~tF).Method] = deal("unique");

                    case "randomsearch"

                        %Continuous
                        [parameters(tF).Method] = deal("rand");
                        [parameters(tF).Samples] = deal(obj.Evaluations);

                        %Discrete
                        [parameters(~tF).Method] = deal("rand");
                        [parameters(~tF).Samples] = deal( obj.Evaluations );

                end %switch

                for iObj = 1 : numel( parameters )

                    param = parameters( iObj );

                    results{iObj} = param.sample();
                    levels{iObj}  = 1 : numel( results{iObj} );

                end

                result = struct();
                switch obj.Search
                    case "gridsearch"

                        sampleArray = combvec( levels{:} );

                        nSample = size(sampleArray, 2);

                        for iObj = 1 : numel( parameters )

                            name = parameters(iObj).Name;

                            if iscell( results{iObj}( sampleArray(iObj,:) ) )
                                values = results{iObj}( sampleArray(iObj,:) );
                            else
                                values = num2cell( results{iObj}( sampleArray(iObj,:) ) );            
                            end  %if iscell

                            [result(1:nSample).(name)] = deal( values{:} );

                        end

                    case "randomsearch"

                        nSample = obj.Evaluations;

                        for iObj = 1 : numel( parameters )
                            name = parameters(iObj).Name;

                            if iscell( results{iObj} )
                                values = results{iObj};
                            else
                                values = num2cell( results{iObj} );
                            end

                            [result(1:nSample).(name)] = deal( values{:} );
                        end

                end

                obj.Samples = result;
      
            end %~isempty( obj.PipelineConfiguration )

            
            obj.Trials = table( 'Size', [obj.nSamples, 2], 'VariableTypes', ["cell" "logical"],...
                'VariableNames', ["Record" "Prepared"], ...
                'RowNames', "Trial" + (1:obj.nSamples) );
            
            [obj.Trials.Record(:)] = deal( {struct()} );

            if ~isempty( obj.AdditionalConfiguration )
                obj.i_buildconstants();
            end

            
        end %sample
        
 
        function value = preview( obj )
            %Preview Preview experiment design/runs
            if ~isempty( obj.Samples )
                value = struct2table( obj.Samples, 'AsArray', true );
                value.Properties.RowNames = "Trial" + (1:height(value));
            else
                value = table();
            end
        end %preview
        

         function value = list( obj )
            value = obj.Trials;
         end
         
         
         function value = describe( obj )
             
             if all( obj.Trials.Prepared )
                 items = [obj.Trials.Record{:}];
                 names = table([items.recordName]', 'VariableNames',"Name");
                 value = [names, obj.table];
             else
                 value = [] ; 
             end
             
         end
         
    end
    
    methods
       
        function value = get.nSamples( obj )
            
            if isempty( obj.Samples )
                value = 1;
            else
                value = numel( obj.Samples );
            end
            
        end
            
        
        function value = get.validTrialIndices( obj )
           
            if ~isempty( obj )
                value = find( obj.Trials.Prepared == true );
            else
                value = [];
            end
        end %get.validTrialIndices
        
    end 
    
    methods (Access = private)
       
        function i_buildconstants( obj )
        
            constants = struct();
            nSample = obj.nSamples;
            
            parameters = obj.AdditionalConfiguration;
            for iObj = 1 : numel( parameters )
                name = parameters(iObj).Name;
                
                if iscell( parameters(iObj).Value )
                    values = repmat( parameters(iObj).Value, 1, nSample );
                else
                    values = repmat({parameters(iObj).Value}, 1, nSample );
                end
                
                [constants(1:nSample).(name)] = deal( values{:} );
            end
            
            obj.Constants = constants;
            
        end %i_buildconstants
        
    end %private
    
end %classdef

%Local functions for validation 
function mustBeClass(a,b)
   if class(a) ~= b
      error('Invalid class')
   end
end