classdef Label < matlab.mixin.SetGet
    %LABEL Summary of this class goes here
    
    properties ( SetAccess = protected, GetAccess = public )
        Name = ""
    end
    
    properties
       label
       pipe
       metadata
    end
    
    methods
        function obj = Label( varargin )
            %LABEL Construct an instance of this class
            if ~isempty( varargin )
               set(obj, varargin{:}) 
            end
        end
    end %constructor 
        
    methods      
        function value = describe( obj  )
            %describe TODO 
            
            if ~isempty(obj)
                meta =cell(numel(obj),1);
                for i = 1:numel(obj)
                    meta{i} = obj(i).metadata;
                end
                value = vertcat( meta{:} );
            else 
                value = table(); 
            end
                
        end %function 
    end
    
    methods (Static)
        function value = default( options )
            
            arguments
               options.ModelType(1,1) string 
                
            end
            
            switch options.ModelType
 
                case "Unsupervised"
                    modelType = "Cluster Empty";
                    nClusters = nan;
                    
                    metadata = table(modelType, nClusters);
                    
                    value = mlp.Label( 'metadata', metadata );           
            end
        end
    end
   
end %classdef 

