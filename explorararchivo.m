% tet5_final.m - VERSIÓN FINAL CORREGIDA
% Todas las variables se pasan correctamente a las funciones

close all; clear; clc;

%% 1. OPCIÓN DE CARGAR MODELO
disp('¿Deseas cargar un modelo entrenado en .mat?');
choice = input('Escribe 1 para cargar modelo, 0 para entrenar desde cero: ');

if choice == 1
    [file, path] = uigetfile('*.mat', 'Selecciona archivo .mat de modelo entrenado');
    if isequal(file, 0)
        disp('No se seleccionó ningún archivo. Se entrenará desde cero');
        error('Para esta versión, necesitas cargar un modelo pre-entrenado');
    else
        modeloPath = fullfile(path, file);
        disp(['Cargando modelo desde: ', modeloPath]);
        
        % Cargar TODAS las variables del archivo
        try
            modeloData = load(modeloPath);
            
            % Extraer variables específicas
            if isfield(modeloData, 'cnnNet')
                cnnNet = modeloData.cnnNet;
                disp('✓ CNN cargada correctamente');
            else
                error('El archivo no contiene la variable cnnNet');
            end
            
            if isfield(modeloData, 'mlpModel')
                mlpModel = modeloData.mlpModel;
                disp('✓ MLP cargado correctamente');
            else
                error('El archivo no contiene la variable mlpModel');
            end
            
            if isfield(modeloData, 'classNames')
                classNames = modeloData.classNames;
            else
                classNames = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];
            end
            
            if isfield(modeloData, 'imageSize')
                imageSize = modeloData.imageSize;
            else
                imageSize = [48 48];
            end
            
            disp('✓ Modelo cargado exitosamente');
            modeloCargado = true;
            
        catch ME
            error('Error al cargar el modelo: %s', ME.message);
        end
    end
else
    error('Para esta versión, selecciona 1 para cargar modelo pre-entrenado');
end

%% 2. CONFIGURACIÓN
projectRoot = pwd;
inputSize = [imageSize 1];

% Verificar que existan las carpetas de emociones (solo para referencia)
disp('=== VERIFICANDO ESTRUCTURA ===');
foldersOK = true;
for i = 1:length(classNames)
    folderPath = fullfile(projectRoot, classNames(i));
    if exist(folderPath, 'dir')
        imgCount = length(dir(fullfile(folderPath, '*.png'))) + ...
                  length(dir(fullfile(folderPath, '*.jpg'))) + ...
                  length(dir(fullfile(folderPath, '*.jpeg')));
        disp(['  ' classNames{i} ': ' num2str(imgCount) ' imágenes']);
    else
        disp(['  ' classNames{i} ': No encontrada (ok para interfaz)']);
    end
end

% Carpeta de resultados
resultsFolder = fullfile(projectRoot, 'resultados');
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
    disp('✓ Carpeta resultados creada');
end

%% 3. INTERFAZ DE CÁMARA
disp('=== INTERFAZ DE CÁMARA ===');

% Variables compartidas (usando appdata para evitar problemas de scope)
setappdata(0, 'stopCamera', false);
setappdata(0, 'photoCount', 0);
setappdata(0, 'cnnNet', cnnNet);
setappdata(0, 'mlpModel', mlpModel);
setappdata(0, 'imageSize', imageSize);
setappdata(0, 'resultsFolder', resultsFolder);

% Verificar cámaras disponibles
try
    camList = webcamlist;
    if isempty(camList)
        error('No se detectaron cámaras');
    end
    disp(['Cámaras disponibles: ' strjoin(camList', ', ')]);
    camName = camList{1};
catch
    camName = 'Integrated Camera';
    disp('Usando cámara por defecto');
end

try
    % ========== INICIALIZAR CÁMARA ==========
    disp('Inicializando cámara...');
    cam = [];
    
    % Intentar con diferentes configuraciones
    resolutions = {'320x240', '640x480', '1280x720'};
    for res = resolutions
        try
            cam = webcam(camName);
            cam.Resolution = res{1};
            testFrame = snapshot(cam);
            fprintf('✓ Cámara inicializada: %s (%dx%d)\n', res{1}, size(testFrame, 2), size(testFrame, 1));
            break;
        catch
            if ~isempty(cam)
                clear cam;
                cam = [];
            end
        end
    end
    
    if isempty(cam)
        % Último intento sin especificar resolución
        try
            cam = webcam(camName);
            testFrame = snapshot(cam);
            fprintf('✓ Cámara inicializada: Resolución por defecto (%dx%d)\n', ...
                   size(testFrame, 2), size(testFrame, 1));
        catch ME
            error('No se pudo inicializar la cámara: %s', ME.message);
        end
    end
    
    % Guardar cámara en appdata
    setappdata(0, 'cam', cam);
    
    % ========== CONFIGURAR DETECTOR DE ROSTROS ==========
    faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
    setappdata(0, 'faceDetector', faceDetector);
    
    % ========== CREAR INTERFAZ GRÁFICA ==========
    disp('Creando interfaz gráfica...');
    
    hFig = figure('Name', 'Detección de Emociones - Tiempo Real', ...
                 'NumberTitle', 'off', ...
                 'Position', [100, 100, 800, 650], ...
                 'MenuBar', 'none', ...
                 'ToolBar', 'none');
    
    % Área para mostrar video
    hAx = axes('Parent', hFig, 'Units', 'normalized', ...
              'Position', [0.05, 0.15, 0.9, 0.75]);
    hImg = imshow(zeros(480, 640, 3, 'uint8'), 'Parent', hAx);
    title(hAx, 'Detección de Emociones Faciales', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Panel de control
    uicontrol('Style', 'text', ...
             'String', 'Estado: ACTIVO', ...
             'Position', [20, 20, 150, 30], ...
             'FontSize', 11, ...
             'FontWeight', 'bold', ...
             'BackgroundColor', [0.2, 0.8, 0.2], ...
             'ForegroundColor', [1, 1, 1]);
    
    uicontrol('Style', 'text', ...
             'String', 'Presiona ESC para salir', ...
             'Position', [200, 20, 200, 30], ...
             'FontSize', 10, ...
             'BackgroundColor', [0.8, 0.8, 0.8]);
    
    % Contador de FPS
    fpsText = uicontrol('Style', 'text', ...
                       'String', 'FPS: --', ...
                       'Position', [450, 20, 100, 30], ...
                       'FontSize', 10);
    
    % Botón para capturar foto
    uicontrol('Style', 'pushbutton', ...
             'String', 'Capturar Foto', ...
             'Position', [600, 20, 120, 30], ...
             'FontSize', 10, ...
             'Callback', @capturePhotoCallback);
    
    % ========== CONFIGURAR CALLBACKS ==========
    set(hFig, 'CloseRequestFcn', @closeAppCallback);
    set(hFig, 'KeyPressFcn', @keyPressCallback);
    
    % ========== VARIABLES DE CONTROL ==========
    frameCount = 0;
    tStart = tic;
    lastProcessTime = 0;
    processEveryNFrames = 3;  % Procesar cada 3 frames para equilibrio velocidad/precisión
    
    % Cache para emociones detectadas
    lastEmotion = '';
    lastBbox = [];
    frameCounter = 0;
    
    % ========== BUCLE PRINCIPAL ==========
    disp('Iniciando detección en tiempo real...');
    disp('Instrucciones:');
    disp('  1. Mira hacia la cámara');
    disp('  2. Asegúrate de tener buena iluminación');
    disp('  3. Presiona ESC para salir');
    disp('  4. Botón "Capturar Foto" para guardar imágenes');
    
    % Obtener variables del appdata para uso local
    cnnNet_local = getappdata(0, 'cnnNet');
    mlpModel_local = getappdata(0, 'mlpModel');
    imageSize_local = getappdata(0, 'imageSize');
    stopCamera = false;
    
    while ~stopCamera && ishandle(hFig)
        try
            % Capturar frame
            frame = snapshot(cam);
            frameCount = frameCount + 1;
            frameCounter = frameCounter + 1;
            
            % Guardar frame actual
            setappdata(0, 'currentFrame', frame);
            
            % Decidir si procesar este frame
            shouldProcess = (frameCounter >= processEveryNFrames) || isempty(lastBbox);
            frameOut = frame;
            
            if shouldProcess
                frameCounter = 0;
                
                % Detectar rostros
                grayFrame = rgb2gray(frame);
                bboxes = step(faceDetector, grayFrame);
                
                if ~isempty(bboxes)
                    % Tomar la cara más grande
                    areas = bboxes(:,3) .* bboxes(:,4);
                    [~, idxMax] = max(areas);
                    bbox = bboxes(idxMax, :);
                    lastBbox = bbox;
                    
                    % Preprocesar rostro
                    faceCrop = imcrop(frame, bbox);
                    faceGray = imresize(rgb2gray(faceCrop), imageSize_local);
                    faceSingle = im2single(faceGray);
                    faceSingle = reshape(faceSingle, [imageSize_local 1]);
                    
                    % Predecir emoción (USANDO VARIABLES LOCALES)
                    featReal = activations(cnnNet_local, faceSingle, 'fc_features', 'OutputAs', 'rows');
                    predReal = mlpModel_local.predict(featReal);
                    lastEmotion = char(predReal);
                    
                    % Dibujar en el frame
                    frameOut = drawEmotionOnFrame(frame, bbox, lastEmotion);
                else
                    lastBbox = [];
                    lastEmotion = '';
                    
                    % Mostrar mensaje
                    [h, w, ~] = size(frame);
                    frameOut = insertText(frame, [w/2-100, 50], ...
                                         'Buscando rostros...', ...
                                         'FontSize', 16, ...
                                         'BoxColor', 'black', ...
                                         'TextColor', 'white', ...
                                         'AnchorPoint', 'Center');
                end
            else
                % Usar la última detección
                if ~isempty(lastBbox) && ~isempty(lastEmotion)
                    frameOut = drawEmotionOnFrame(frame, lastBbox, lastEmotion);
                end
            end
            
            % Calcular FPS
            elapsedTime = toc(tStart);
            if elapsedTime > 1
                fps = frameCount / elapsedTime;
                set(fpsText, 'String', sprintf('FPS: %.1f', fps));
                frameCount = 0;
                tStart = tic;
            end
            
            % Actualizar display
            set(hImg, 'CData', frameOut);
            drawnow limitrate;
            
        catch ME
            if ~ishandle(hFig)
                break;
            end
            warning('Error en frame %d: %s', frameCount, ME.message);
            continue;
        end
        
        % Verificar si se debe detener
        stopCamera = getappdata(0, 'stopCamera');
    end
    
    % ========== LIMPIAR RECURSOS ==========
    disp('Cerrando aplicación...');
    
    if ~isempty(cam) && isvalid(cam)
        clear cam;
        disp('✓ Cámara liberada');
    end
    
    if ishandle(hFig)
        close(hFig);
    end
    
    disp('✓ Interfaz cerrada correctamente');
    
catch ME
    warning('Error en interfaz principal: %s', ME.message);
    disp('Iniciando modo de prueba con imagen estática...');
    testWithStaticImage();
end

disp('=== EJECUCIÓN COMPLETADA ===');

%% ========== FUNCIONES ==========

% Función para dibujar emoción en el frame
function frameOut = drawEmotionOnFrame(frame, bbox, emotion)
    % Mapeo de colores
    colorMap = containers.Map();
    colorMap('angry') = [255, 0, 0];
    colorMap('disgust') = [128, 0, 128];
    colorMap('fear') = [128, 128, 0];
    colorMap('happy') = [0, 255, 0];
    colorMap('sad') = [0, 0, 255];
    colorMap('surprise') = [255, 165, 0];
    colorMap('neutral') = [128, 128, 128];
    
    if isKey(colorMap, emotion)
        boxColor = colorMap(emotion);
    else
        boxColor = [0, 255, 0];
    end
    
    % Dibujar bounding box
    frameOut = insertShape(frame, 'Rectangle', bbox, ...
                          'Color', boxColor, ...
                          'LineWidth', 3);
    
    % Agregar texto
    textPos = [bbox(1), bbox(2)-35];
    frameOut = insertText(frameOut, textPos, ...
                         emotion, ...
                         'FontSize', 14, ...
                         'BoxColor', boxColor, ...
                         'TextColor', 'white', ...
                         'BoxOpacity', 0.8);
end

% Función para cerrar la aplicación
function closeAppCallback(~, ~)
    setappdata(0, 'stopCamera', true);
    
    % Liberar cámara si existe
    if isappdata(0, 'cam')
        cam = getappdata(0, 'cam');
        if ~isempty(cam) && isvalid(cam)
            clear cam;
            rmappdata(0, 'cam');
        end
    end
    
    fprintf('✓ Aplicación cerrada\n');
    delete(gcf);
end

% Callback para tecla ESC
function keyPressCallback(~, event)
    if strcmp(event.Key, 'escape')
        setappdata(0, 'stopCamera', true);
        disp('✓ Tecla ESC presionada. Cerrando...');
    end
end

% Callback para capturar foto - CORREGIDO
function capturePhotoCallback(~, ~)
    try
        % Obtener todas las variables necesarias del appdata
        cam = getappdata(0, 'cam');
        cnnNet = getappdata(0, 'cnnNet');
        mlpModel = getappdata(0, 'mlpModel');
        imageSize = getappdata(0, 'imageSize');
        faceDetector = getappdata(0, 'faceDetector');
        resultsFolder = getappdata(0, 'resultsFolder');
        
        if isempty(cam) || ~isvalid(cam)
            error('Cámara no disponible');
        end
        
        % Obtener contador de fotos
        if isappdata(0, 'photoCount')
            photoCount = getappdata(0, 'photoCount') + 1;
        else
            photoCount = 1;
        end
        setappdata(0, 'photoCount', photoCount);
        
        % Tomar una nueva foto
        disp('Capturando foto...');
        frame = snapshot(cam);
        
        % Procesar para detectar rostro
        grayFrame = rgb2gray(frame);
        bboxes = step(faceDetector, grayFrame);
        
        if ~isempty(bboxes)
            % Tomar la cara más grande
            areas = bboxes(:,3) .* bboxes(:,4);
            [~, idxMax] = max(areas);
            bbox = bboxes(idxMax, :);
            
            % Preprocesar y predecir
            faceCrop = imcrop(frame, bbox);
            faceGray = imresize(rgb2gray(faceCrop), imageSize);
            faceSingle = im2single(faceGray);
            faceSingle = reshape(faceSingle, [imageSize 1]);
            
            featReal = activations(cnnNet, faceSingle, 'fc_features', 'OutputAs', 'rows');
            predReal = mlpModel.predict(featReal);
            emotion = char(predReal);
            
            % Dibujar en el frame
            frameOut = drawEmotionOnFrame(frame, bbox, emotion);
            
            % Nombre del archivo
            photoName = sprintf('emocion_%s_%03d.jpg', emotion, photoCount);
            photoPath = fullfile(resultsFolder, photoName);
            
        else
            % Sin rostro detectado
            frameOut = frame;
            emotion = 'desconocida';
            photoName = sprintf('sin_rostro_%03d.jpg', photoCount);
            photoPath = fullfile(resultsFolder, photoName);
        end
        
        % Guardar la imagen
        imwrite(frameOut, photoPath);
        fprintf('✓ Foto guardada: %s (Emoción: %s)\n', photoName, emotion);
        
        % Mostrar mensaje
        msg = sprintf('Foto #%d guardada\n%s\nEmoción: %s', ...
                     photoCount, photoName, emotion);
        msgbox(msg, 'Captura Exitosa', 'help');
        
        % Mostrar miniatura
        hFig2 = figure('Name', 'Foto Capturada', ...
                      'NumberTitle', 'off', ...
                      'Position', [900, 300, 400, 400]);
        imshow(frameOut);
        title(sprintf('Captura #%d: %s', photoCount, emotion), 'FontSize', 12);
        
    catch ME
        warning('Error al capturar foto: %s', ME.message);
        errordlg(sprintf('Error al capturar foto:\n%s', ME.message), 'Error');
    end
end

% Función para prueba con imagen estática
function testWithStaticImage()
    try
        % Obtener variables del appdata
        cnnNet = getappdata(0, 'cnnNet');
        mlpModel = getappdata(0, 'mlpModel');
        imageSize = getappdata(0, 'imageSize');
        classNames = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];
        projectRoot = pwd;
        
        disp('Buscando imagen de prueba...');
        
        for emotion = classNames
            folderPath = fullfile(projectRoot, char(emotion));
            if exist(folderPath, 'dir')
                imageFiles = dir(fullfile(folderPath, '*.jpg'));
                if isempty(imageFiles)
                    imageFiles = dir(fullfile(folderPath, '*.png'));
                end
                
                if ~isempty(imageFiles)
                    imgIdx = randi(length(imageFiles));
                    imgPath = fullfile(folderPath, imageFiles(imgIdx).name);
                    
                    fprintf('Probando con: %s\n', imageFiles(imgIdx).name);
                    
                    img = imread(imgPath);
                    faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
                    grayImg = rgb2gray(img);
                    bboxes = faceDetector(grayImg);
                    
                    if ~isempty(bboxes)
                        bbox = bboxes(1, :);
                        faceCrop = imcrop(img, bbox);
                        faceGray = imresize(rgb2gray(faceCrop), imageSize);
                        faceSingle = im2single(faceGray);
                        faceSingle = reshape(faceSingle, [imageSize 1]);
                        
                        featReal = activations(cnnNet, faceSingle, 'fc_features', 'OutputAs', 'rows');
                        predReal = mlpModel.predict(featReal);
                        
                        figure('Name', 'Prueba Estática', ...
                              'NumberTitle', 'off', ...
                              'Position', [200, 200, 800, 600]);
                        
                        imshow(img); hold on;
                        rectangle('Position', bbox, 'EdgeColor', [0, 1, 0], 'LineWidth', 3);
                        text(bbox(1), bbox(2)-25, ['Emoción: ' char(predReal)], ...
                             'Color', 'white', 'BackgroundColor', 'black', ...
                             'FontSize', 14, 'FontWeight', 'bold');
                        
                        title(sprintf('Imagen: %s\nEmoción detectada: %s', ...
                              imageFiles(imgIdx).name, char(predReal)), 'FontSize', 16);
                        
                        fprintf('✓ Emoción detectada: %s\n', char(predReal));
                        return;
                    end
                end
            end
        end
        
        disp('No se encontraron imágenes de prueba adecuadas');
        
    catch ME
        warning('Error en prueba estática: %s', ME.message);
    end
end