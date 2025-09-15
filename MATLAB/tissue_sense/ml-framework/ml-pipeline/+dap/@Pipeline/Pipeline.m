classdef Pipeline < dap.Base
    %PIPELINE Create a data analysis pipeline 
    %
    %
    % dap.Pipeline methods:
    %
    
    
    
    
    %   Copyright 2019 The MathWorks, Inc.
    %
    % Auth/Revision:
    %   MathWorks Consulting
    %   $Author: nhowes $
    %   $Revision: 324 $  $Date: 2019-10-29 13:16:05 -0400 (Tue, 29 Oct 2019) $
    % ---------------------------------------------------------------------
    

    methods
        
        function obj = Pipeline( data, varargin )
            %Pipeline Construct an instance of this class
            
            if nargin > 0  
                obj.Source = data; 
                obj.Data   = data;
                
                if ~isempty( varargin )
                    set( obj, varargin{:} )
                end
                    
            end
            
        end %Pipeline

        function adddata( obj, value )
            
            for iObj = 1:numel(obj)
                obj(iObj).Data = value;
                obj(iObj).Source = value;
                obj(iObj).reset();
            end
            
        end %function
        
        
       function varargout = update( obj, functions, varargin )
           
           varargout = cell(1, nargout);
           
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
                   
                   newRecord = dap.stepInPipeline.Record( ...
                       'Operation', functions );
                   
                    obj.addStep( newRecord );
                    
               end
           else
               error('Result is not a timetable')
           end
           
           
            if exist( 'info', 'var' )    
                varargout{1} = info;
            end
           
       end %update  
       
     
       
       function result = vars( obj, list, rule )
           %VARS
           
           arguments
               obj
               list (1,:) {stringorhandleValidation( list ) } = obj.VariableNames
               rule (1,1) = "matches"
           end
           
           if isa( list, 'function_handle' )
               tF           = varfun(list, obj.Data, "OutputFormat", "uni");
               result       = obj.VariableNames(tF);
           elseif isstring( list )
               switch rule
                   case "matches"
                       tF = matches( obj.VariableNames, list );
                   case "contains"
                       tF = contains( obj.VariableNames, list );
                   case "startsWith"
                       tF = startsWith( obj.VariableNames, list );
                   case "endsWith"
                       tF = endsWith( obj.VariableNames, list );
               end
               result = obj.VariableNames(tF);
           end
           
       end %vars
       
       
    end %public
        
    methods ( Access = protected )
        function setPrior( obj )
            obj.Prior = obj.Data;
        end
        
        function addStep(obj, newStep)
            obj.Step = [obj.Step newStep];
        end
    end %protected
 
    methods (Static)
       
        function value = new( varargin )
            %NEW
            value = dap.Pipeline( varargin{:} );
        end %new
        
        function value = restore( data ,steps )
           %RESTORE
           
            value = dap.Pipeline( data );
            
            nSteps = numel( steps );
            for iStep = 1 : nSteps  
                steps( iStep ).restore( value )
            end
                
        end %restore     
        
        function value = load( data, location )
            %LOAD
            
            importedSteps = load( location );
            
            steps = importedSteps.pipelineSteps; 
            
            value = dap.Pipeline.restore( data, steps );
            
        end %load
        
    end %static
  
end %classdef

% Custom validator functions
function stringorhandleValidation(input)
    % Test for specific class
    if ~isstring( input ) && ~isa( input, 'function_handle' )
        error('Input must be a function handle or string.')
    end
end 

