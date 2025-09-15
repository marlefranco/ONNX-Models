classdef som < matlab.mixin.SetGet
    %SOM Self-organizing map. 
    %
    % Syntax:
    %
    %   obj.som( data )
    %
    %   obj.som( data, dimensions, coverSteps, initNeighbor, topologyFcn, distanceFcn )
    %
    
    properties
        data         (:,:) double     
        clusters     (:,1) double
        dimensions   (1,2) double = [2 2]; 
        coverSteps   (1,1) double = 100
        initNeighbor (1,1) double = 3
        topologyFcn  (1,1) string ...
            {mustBeMember(topologyFcn,["hextop", "tritop", "gridtop", "randtop"])} = "hextop";
        distanceFcn  (1,1) string ...
            {mustBeMember(distanceFcn,["linkdist", "dist", "mandist", "negdist", "linkdist"])} = "linkdist"   
    end
    
    properties
        net 
    end
    
    methods
        function obj = som( varargin )
            %SOM Construct an instance of this class
    
            if nargin > 0 
            
                obj.iParseInputArguments( varargin{:} )
            
                x = transpose( obj.data );

                % Create a Self-Organizing Map
                net = selforgmap( obj.dimensions, ...
                        obj.coverSteps, ...
                        obj.initNeighbor, ...
                        obj.topologyFcn, ...
                        obj.distanceFcn ...
                        );

                net.trainParam.showWindow = false;
                
                % Train the Network
                [obj.net,~] = train(net, x );

                % Test the Network
                y = obj.net( x );

                % Transofmr Vectors 
                obj.clusters = transpose( vec2ind(y) );
            
            end
        end
        
        function iParseInputArguments( obj, varargin )
            
            if istable( varargin{1} ) || istimetable( varargin{1} )
                obj.data = varargin{1}.Variables;
            else
                obj.data = varargin{1};
            end
    
            if nargin > 2
               obj.dimensions = varargin{2};
            end
            
            if nargin > 3
                obj.coverSteps = varargin{3};
            end
            
            if nargin > 4
                obj.initNeighbor = varargin{4};
            end
            
            if nargin > 5
                obj.topologyFcn = varargin{5};
            end
            
            if nargin > 6
                obj.distanceFcn = varargin{6};
            end
            
        end
    end
end %classdef

