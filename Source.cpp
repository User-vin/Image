#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <C:/Users/scott/OneDrive/Bureau/opencv/json.hpp>//télécharger le fichier json nlohmann (pas besoin d'installation, juste include ce fichier)

using json = nlohmann::json;

//utilisation de labelme pour générer les json (avec des polygones à quatres points, pas des rectangles)

//Fonction qui permet de savoir si deux boites englobantes sont proches ou si leur intersection n'est pas vide
bool is_close_or_overlapping(cv::Rect bbox1, cv::Rect bbox2, int distance_threshold = 10, int small_area_threshold = 300, double overlap_threshold = 0.3) {
	int intersection_x1 = cv::max(bbox1.x, bbox2.x);
	int intersection_y1 = cv::max(bbox1.y, bbox2.y);
	int intersection_x2 = cv::min(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
	int intersection_y2 = cv::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);

	int intersection_area = cv::max(0, intersection_x2 - intersection_x1) * cv::max(0, intersection_y2 - intersection_y1);

	int area1 = bbox1.area();
	int area2 = bbox2.area();

	double overlap_ratio = static_cast<double>(intersection_area) / (area1 + area2 - intersection_area);

	bool horizontal_close = (bbox2.x <= bbox1.x + bbox1.width + distance_threshold) && (bbox1.x <= bbox2.x + bbox2.width + distance_threshold);
	bool vertical_close = (bbox2.y <= bbox1.y + bbox1.height + distance_threshold) && (bbox1.y <= bbox2.y + bbox2.height + distance_threshold) && (area1 < small_area_threshold || area2 < small_area_threshold);

	return (horizontal_close && vertical_close) || (overlap_ratio >= overlap_threshold);
}

//Fonction qui permet de savoir si une boite en contient une autre (totalement ou presque, selon un seuil)
bool is_contained(cv::Rect bbox1, cv::Rect bbox2, double threshold = 0.9) {
	int intersection_x1 = cv::max(bbox1.x, bbox2.x);
	int intersection_y1 = cv::max(bbox1.y, bbox2.y);
	int intersection_x2 = cv::min(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
	int intersection_y2 = cv::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);

	int intersection_area = cv::max(0, intersection_x2 - intersection_x1) * cv::max(0, intersection_y2 - intersection_y1);
	int bbox1_area = bbox1.area();

	double overlap_ratio = static_cast<double>(intersection_area) / bbox1_area;

	return overlap_ratio >= threshold;
}


//Fonction qui permet de trouver des minimums locaux dans un tableau
std::vector<int> find_local_minima(cv::Mat mat, int range) {

	std::vector<int> v;

	//Flou gaussien sur mat (l'histogramme projeté vertical)
	int kernel_size = 5;
	double sigma = 1.67;
	cv::GaussianBlur(mat, mat, cv::Size(kernel_size, 1), sigma, sigma);

	const unsigned char* mat_data = mat.ptr<unsigned char>();

	for (int i = 0; i < mat.cols; i++) {
		v.push_back(static_cast<int>(mat_data[i]));
	}

	std::vector<int> minima;

	//Parcours du vecteur pour trouver les minimums locaux
	for (int i = range; i < v.size() - range; i++) {
		bool is_min = true;

		//On regarde si l'élément courant est un minimum local
		for (int j = i - range; j <= i + range; j++) {
			if (v[j] < v[i]) {
				is_min = false;
				break;
			}
		}

		//Si c'est un minimum local, on l'ajoute à la liste des minimums locaux
		if (is_min) {
			minima.push_back(i);
		}
	}
	return minima;
}


//Fonction qui permet de tracer des lignes noires sur une image
//En fonction des minimums trouvés sur son histogramme projeté vertical
void set_values_to_zero(cv::Mat& mat, const std::vector<int>& indices) {
	for (int i = 0; i < indices.size(); i++) {
		int col_index = indices[i];
		if (col_index >= 0 && col_index < mat.cols) {
			for (int j = 0; j < mat.rows; j++) {
				mat.at<char>(j, col_index) = 0;
			}
		}
	}
}



//Fonction principale permettant de faire la détéction du tableau, des lignes, des mots et des lettres
std::tuple<json, cv::Mat, cv::Mat, int, int, float> detection(std::string path, std::string name) {

	//Conversion en hsv -> flou gaussien -> seuillage (sélection des pixels verts)
	cv::Mat hsv, gaussian, binary, closing, img = cv::imread(path);
	cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

	cv::GaussianBlur(hsv, gaussian, cv::Size(3, 3), 0);
	cv::inRange(gaussian, cv::Scalar(30, 0, 0), cv::Scalar(120, 255, 170), binary);

	//Fermeture binaire
	cv::Mat kernel = cv::Mat::ones(cv::Size(50, 50), CV_8UC1);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::morphologyEx(binary, closing, cv::MORPH_OPEN, kernel);


	//Recherche des contours du tableau
	cv::findContours(closing, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	
	//Sélection de la plus grande composante connexe (le tableau)
	double largest_area = 0;
	int largest_area_index = 0;
	for (int i = 0; i < contours.size(); i++) {
		double area = cv::contourArea(contours[i]);
		if (area > largest_area) {
			largest_area = area;
			largest_area_index = i;
		}
	}
	
	//Approximation du polygone correspondant au contour trouvé par un quadrilatère (plus facile à gérer pour la suite)
	std::vector<cv::Point> approx;
	double epsilon = 0.1 * cv::arcLength(contours[largest_area_index], true);

	while (true) {
		cv::approxPolyDP(contours[largest_area_index], approx, epsilon, true);
		if (approx.size() == 4) {
			break;
		}
		epsilon *= 0.9;
	}

	std::vector<std::vector<cv::Point>> contours_poly(1);
	contours_poly[0] = approx;

	//On récupère les coordonnées de la boite englobante pour le crop + warp
	cv::Point p0 = contours_poly[0][0];
	cv::Point p1 = contours_poly[0][1];
	cv::Point p2 = contours_poly[0][2];
	cv::Point p3 = contours_poly[0][3];

	int x0 = p0.x, y0 = p0.y;
	int x1 = p1.x, y1 = p1.y;
	int x2 = p2.x, y2 = p2.y;
	int x3 = p3.x, y3 = p3.y;

	int x_min = std::min({x0,x1,x2,x3});
	int y_min = std::min({y0,y1,y2,y3});
	int x_max = std::max({ x0,x1,x2,x3 });
	int y_max = std::max({ y0,y1,y2,y3 });

	//Crop du tableau
	cv::Rect bounding_rect = cv::boundingRect(approx);
	cv::Mat cropped_image = img(bounding_rect);

	//Calcul des moments de la forme
	cv::Moments moments = cv::moments(contours[largest_area_index]);

	//Calcul du centroid avec les moments
	cv::Point2f centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

	//Tri des points du polygone (tableau) dans le sens horaire
	std::sort(approx.begin(), approx.end(), [centroid](cv::Point a, cv::Point b) {
		return std::atan2(a.y - centroid.y, a.x - centroid.x) < std::atan2(b.y - centroid.y, b.x - centroid.x);
		});

	//Vecteur contenant les 4 points du tableau
	std::vector<cv::Point2f> src_points;
	src_points.push_back(approx[0]);
	src_points.push_back(approx[1]);
	src_points.push_back(approx[2]);
	src_points.push_back(approx[3]);

	//Vecteur contenant les 4 nouveaux points du tableau pour le warp
	std::vector<cv::Point2f> dst_points;
	dst_points.push_back(cv::Point2f(0, 0));
	dst_points.push_back(cv::Point2f(img.cols, 0));
	dst_points.push_back(cv::Point2f(img.cols, img.rows));
	dst_points.push_back(cv::Point2f(0, img.rows));

	//Matrice de transformation pour le warp créée à partir des points de départ et d'arrivée
	cv::Mat transform_matrix = cv::getPerspectiveTransform(src_points, dst_points);
	cv::Mat warped_board;

	//Warp
	cv::warpPerspective(img, warped_board, transform_matrix, cv::Size(img.cols, img.rows));

	//Création de l'objet json qui va contenir tous les labels détectés:
	//"labels" : ["board", "board", "line", "line", "line", "word", "letter", etc]
	//"coords": [[x1, y1, x2, y2, x3, y3, x4, y4], [x1, y1, x2, y2, x3, y3, x4, y4], etc]
	json js = {
		{"labels", json::array()},
		{"coords", json::array()}
	};
	js["coords"].insert(js["coords"].end(), json::array({ src_points[0].x, src_points[0].y, src_points[1].x, src_points[1].y, src_points[2].x, src_points[2].y, src_points[3].x, src_points[3].y }));
	js["labels"].insert(js["labels"].end(), "board");

	//Conversion du tableau croppé en gris
	cv::Mat  gray_board, binary_board, closed_board;
	cv::cvtColor(warped_board, gray_board, cv::COLOR_BGR2GRAY);




	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Seuillage global
	cv::threshold(gray_board, binary_board, 140, 255, cv::THRESH_BINARY);

	//Création du noyau de taille (45,5) pour fermeture binaire horizontale pour coller les lettres faisant partie du même mot
	cv::Mat closing_kernel_word = cv::Mat::ones(cv::Size(45, 5), CV_8UC1);
	cv::morphologyEx(binary_board, closed_board, cv::MORPH_CLOSE, closing_kernel_word);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	


	int hh = gray_board.rows;
	int ww = gray_board.cols;
	double scale = 800.0 / hh;

	cv::resize(gray_board, gray_board, cv::Size(static_cast<int>(scale* ww), 800));
	cv::resize(warped_board, warped_board, cv::Size(static_cast<int>(scale* ww), 800));
	cv::resize(binary_board, binary_board, cv::Size(static_cast<int>(scale* ww), 800));
	cv::resize(closed_board, closed_board, cv::Size(static_cast<int>(scale* ww), 800));
	
	//Seuillage local sur le tableau
	cv::Mat thresh;
	cv::adaptiveThreshold(gray_board, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 21, 7);

	//Fermeture binaire sur le tableau
	cv::Mat kernell = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
	cv::Mat morph;
	cv::morphologyEx(thresh, morph, cv::MORPH_CLOSE, kernell);

	//Recherche des composantes connexes
	cv::Mat labels, stats, centroids;
	int num_labels = cv::connectedComponentsWithStats(morph, labels, stats, centroids);

	std::map<int, std::vector<std::vector<cv::Point>>> contours_dict;

	//Recherche des contours après le seuillage, en supprimant tous les contours dont l'aire est inférieure à 100
	for (int label = 1; label < num_labels; ++label) {
		cv::Mat mask = labels == label;
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			contours.erase(remove_if(contours.begin(), contours.end(), [](const std::vector<cv::Point>& cnt) { return cv::contourArea(cnt) <= 100; }), contours.end());
			contours_dict.insert(std::pair<int, std::vector<std::vector<cv::Point>>>(label, contours));
	}

	//Création du vecteur bbox qui va contenir les coordonées
	std::vector<cv::Rect> bboxes;
	for (int label = 1; label < num_labels; ++label) {
		for (const std::vector<cv::Point>& cnt : contours_dict[label]) {
			cv::Rect bbox = boundingRect(cnt);
			bboxes.push_back(bbox);
		}
	}
	/**
	for (auto& box : bboxes) {
		cv::rectangle(warped_board, cv::Rect(box), cv::Scalar(0, 255, 0));
	}
	cv::imwrite("C:/Users/scott/OneDrive/Bureau/images_slides/" + name + "55.jpg", warped_board);
	**/

	//On fusionne les rectangle qui sont très proches ou qui ont une intersection non nulle fixée par un seuil
	std::vector<std::vector<cv::Rect>> clusters;
	for (const cv::Rect& bbox : bboxes) {
		bool added_to_existing_cluster = false;
		for (std::vector<cv::Rect>& cluster : clusters) {
			for (const cv::Rect& other_bbox : cluster) {
				if (is_close_or_overlapping(bbox, other_bbox)) {
					cluster.push_back(bbox);
					added_to_existing_cluster = true;
					break;
				}
			}
			if (added_to_existing_cluster) {
				break;
			}
		}

		if (!added_to_existing_cluster) {
			clusters.push_back(std::vector<cv::Rect>(1, bbox));
		}
	}

	std::vector<cv::Rect> cluster_bboxes;
	int max_area_threshold = 100000;
	double aspect_ratio_threshold = 6.0;
	int max_width_threshold = 400;
	int max_height_threshold = 400;
	int min_dist_border = 20;

	//Suppression des rectangles qui ne respectent pas certaines conditions sur la taille, ou qui sont sur les bords des images
	//(Peu de chance que ça soit de l'écriture)
	for (const std::vector<cv::Rect>& cluster : clusters) {
		int x1 = INT_MAX, y1 = INT_MAX, x2 = 0, y2 = 0;
		for (const cv::Rect& bbox : cluster) {
			x1 = cv::min(x1, bbox.x);
			y1 = cv::min(y1, bbox.y);
			x2 = cv::max(x2, bbox.x + bbox.width);
			y2 = cv::max(y2, bbox.y + bbox.height);
		}

		int area = (x2 - x1) * (y2 - y1);
		int width = x2 - x1;
		int height = y2 - y1;


		int height_warped_board = warped_board.rows;
		int width_warped_board = warped_board.cols;

		double aspect_ratio = static_cast<double>(cv::max(width, height)) / cv::min(width, height);

		if (area <= max_area_threshold && aspect_ratio <= aspect_ratio_threshold && width <= max_width_threshold && height <= max_height_threshold &&
			x1 >= min_dist_border && x1 <= width_warped_board - min_dist_border && y1 >= min_dist_border && y1 <= height_warped_board - min_dist_border) {
			cluster_bboxes.push_back(cv::Rect(x1, y1, width, height));
		}
	}

	//Suppression des rectangles contenus entièrement ou presque dans d'autes rectangles
	std::vector<cv::Rect> filtered_bboxes;
	for (const cv::Rect& bbox1 : cluster_bboxes) {
		bool contained = false;
		for (const cv::Rect& bbox2 : cluster_bboxes) {
			if (bbox1 != bbox2 && is_contained(bbox1, bbox2)) {
				contained = true;
				break;
			}
		}

		if (!contained) {
			filtered_bboxes.push_back(bbox1);
		}
	}

	//Création du vecteur qui va contenir les coordonnées des boites englobantes des mots
	std::vector<std::vector<int>> word_bboxes;
	//Parcours des composantes connexes trouvées (les boites englobantes)
	for(const auto &bbox:filtered_bboxes){

		//On récupère les coordonnées des boites englobantes des mots trouvés
		int x = bbox.x;
		int y = bbox.y;
		int w = bbox.width;
		int h = bbox.height;

		//Condition pour ignorer les composantes connexes trop petites
		if (w*h > 400){
			
			//Dessin du rectangle correspondant au mot courant
			cv::rectangle(warped_board, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);

			//Insertion des coordonnées et label du mot dans le json pour l'évalutaion
			js["coords"].insert(js["coords"].end(), json::array({ x, y, x, y + h, x + w, y + h, x + w, y }));
			js["labels"].insert(js["labels"].end(), "word");

			//Crop du mot
			cv::Mat binary_cropped_closed_word = closed_board(cv::Rect(x, y, w, h));

			//Suppression des lettres des autres mots qui dépassent sur la boite englobante du mot courant
			cv::Mat labs, stats, cent;
			int numLabels = cv::connectedComponentsWithStats(binary_cropped_closed_word, labs, stats, cent);

			//On garde seulement la plus grande composante connexe qui se trouve dans la boite englobante du mot courant (donc le mot lui-même et pas les bouts de lettres des autres mots)
			int largestLabel = 0;
			int largestSize = 0;
			for (int i = 1; i < numLabels; i++) {
				int size = stats.at<int>(i, cv::CC_STAT_AREA);
				if (size > largestSize) {
					largestLabel = i;
					largestSize = size;
				}
			}

			//Application d'un masque pour ne garder que uniquement le mot
			cv::Mat binary_cropped_word = binary_board(cv::Rect(x, y, w, h));
			cv::Mat largestComponentMask = (labs == largestLabel);
			cv::Mat result;
			cv::bitwise_and(binary_cropped_word, largestComponentMask, result);
			//Fin suppression

			//Ajout des bbox des mots dans un vecteur pour pouvoir trouver les lignes (à la fin de la fonction)
			word_bboxes.emplace_back(std::vector<int>{x, y, w, h});

			//Condition pour prendre seulement les boites engloabantes des composantes connexes assez grandes
			int range = 10;
			if (result.cols > range) {

				int width = result.cols;
				int height = 1;

				//Calcul de l'histogramme projeté vertical
				cv::Mat hist = cv::Mat::zeros(height, width, CV_32FC1);
				cv::reduce(result, hist, 0, cv::REDUCE_SUM, CV_32FC1);

				//Normalisation de l'histogramme
				double min_val, max_val;
				cv::minMaxLoc(hist, &min_val, &max_val);
				hist = hist / max_val * 255;
				hist.convertTo(hist, CV_8UC1);

				//Appel de la fonction permettant de trouver les minimums locaux dans l'histogramme projeté
				std::vector<int> minima = find_local_minima(hist, range);

				//Pour chaque minimum trouvé, on trace une ligne noir dans l'image à l'indice correspondant, pour séparer les lettres
				set_values_to_zero(result, minima);

				//Recherche des composantes connexes (les lettres)
				cv::Mat letter_labels, letter_stats, letter_centroids;
				cv::connectedComponentsWithStats(result, letter_labels, letter_stats, letter_centroids);

				//Ajout des labesl et coordonnées correspondantes aux lettres trouvées
				for (int i = 1; i < letter_stats.rows; i++) {
					int xl = letter_stats.at<int>(cv::Point(0, i)) + x;
					int yl = letter_stats.at<int>(cv::Point(1, i)) + y;
					int wl = letter_stats.at<int>(cv::Point(2, i));
					int hl = letter_stats.at<int>(cv::Point(3, i));
					//On ignore les composantes trop petites
					if (wl*hl >= 200) {
						cv::rectangle(warped_board, cv::Rect(xl, yl, wl, hl), cv::Scalar(0, 0, 0));
						js["coords"].insert(js["coords"].end(), json::array({ xl, yl, xl, yl + hl, xl + wl, yl + hl, xl + wl, yl }));
						js["labels"].insert(js["labels"].end(), "letter");
					}
				}
			}
		}
	}

	//Les mots détéctés avec la fonction cv::connectedComponentsWithStats sont triés par défaut en fonction de leur valeur y_min
	//Pour chaque bounding box des mots, on la compare à celle d'avant. Si elle est à peu près à la même hauteur, il appartient à la même ligne
	//Sinon, on crée une nouvelle ligne et on recommence (on peut créer les lignes au fur et à mesure sans retourner en arrière comme les bbox sont déjà triées par y_min
	std::vector<std::vector<int>> lines;
	lines.emplace_back(std::vector<int>{0});



	for (int i = 1; i < word_bboxes.size(); i++) {
		if (((word_bboxes[i - 1][1] + word_bboxes[i - 1][3]) <= word_bboxes[i][1]) || (word_bboxes[i - 1][1] >= (word_bboxes[i][1] + word_bboxes[i][3]))){
			lines.emplace_back(std::vector<int>{i});
		}
		else if ((word_bboxes[i - 1][1] >= word_bboxes[i][1]) && ((word_bboxes[i - 1][1] + word_bboxes[i - 1][3]) <= (word_bboxes[i][1] + word_bboxes[i][3])) || (word_bboxes[i - 1][1] <= word_bboxes[i][1]) && ((word_bboxes[i - 1][1] + word_bboxes[i - 1][3]) >= (word_bboxes[i][1] + word_bboxes[i][3]))) {
			lines.back().push_back(i);
		}
		else if (word_bboxes[i - 1][1] <= word_bboxes[i][1]) {
			int dist = word_bboxes[i - 1][1] + word_bboxes[i - 1][3] - word_bboxes[i][1];
			double vertical_intersection = std::max((double)dist / word_bboxes[i][3], (double)dist / word_bboxes[i-1][3]);
			if (vertical_intersection >= 0.7) {
				lines.back().push_back(i);
			}
			else {
				lines.emplace_back(std::vector<int>{i});
			}
		}
		else if (word_bboxes[i - 1][1] >= word_bboxes[i][1]) {
			int dist = word_bboxes[i][1] + word_bboxes[i][3] - word_bboxes[i - 1][1];
			double vertical_intersection = std::max((double)dist / word_bboxes[i][3], (double)dist / word_bboxes[i-1][3]);
			if (vertical_intersection >= 0.7) {
				lines.back().push_back(i);
			}
			else {
				lines.emplace_back(std::vector<int>{i});
			}
		}

	}


	//tri des mots de chaque ligne en fonction de leur x_min, pour les remettre dans l'ordre
	for (auto& line : lines) {
		std::sort(line.begin(), line.end(), [&](int i, int j) {
			return word_bboxes[i][0] < word_bboxes[j][0];
		});
	}

	//dessin des rectangles des lignes
	//Recherche du x_min du premier élément de la ligne courante, x_max du dernier élément de la ligne courante, et y_max de l'élément le plus 'bas'
	//et y_min de l'élément le plus 'haut' pour dessiner un rectangle qui englobe bien tous les mots dans la ligne
	for (const auto& subvector : lines) {
		int xmin_line = INT_MAX;
		int ymin_line = INT_MAX;
		int ymax_line = INT_MIN;

		for (const auto& val : subvector) {
			xmin_line = std::min(xmin_line, word_bboxes[val][0]);
			ymin_line = std::min(ymin_line, word_bboxes[val][1]);
			ymax_line = std::max(ymax_line, word_bboxes[val][1] + word_bboxes[val][3]);
		}

		int w_line = word_bboxes[subvector.back()][0] + word_bboxes[subvector.back()][2] - xmin_line;
		int h_line = ymax_line - ymin_line;

		//Dessin du rectangle sur l'image
		cv::rectangle(warped_board, cv::Rect(xmin_line, ymin_line, w_line, h_line), cv::Scalar(0, 255, 0));

		//Ajout des coordonnées des lignes dans le json pour l'évaluation
		js["coords"].insert(js["coords"].end(), json::array({ xmin_line, ymin_line, xmin_line, ymin_line + h_line, xmin_line + w_line, ymin_line + h_line, xmin_line + w_line, ymin_line }));
		js["labels"].insert(js["labels"].end(), "line");
	}

	//Sauvegarde du l'image avec les boites englobantes des objets détéctés
	cv::imwrite("C:/Users/scott/OneDrive/Bureau/output2/" + name + ".jpg", warped_board);

	//Renvoie le json des objets détéctés, la matrice de transformation pour l'appliquer sur les annotations, et les dimensions du tableau transformé (warp)
	return std::make_tuple(js, transform_matrix, warped_board, warped_board.cols, warped_board.rows, scale);
}


//Simplifie le json généré par labelme donné en paramètre (enlève toutes les informations inutiles)
//Et application du warp sur les annotations pour que ça corresponde au objets détéctés
json simplify_json_and_transformation(std::string path, const cv::Mat& transform_matrix, int img_width, int img_height, cv::Mat warped_board, float scale) {
	//Lecture du json des annotations
	std::ifstream f(path);
	json js_read = json::parse(f);

	//Création du nouveau json
	json js = {
		{ "labels", json::array() },
		{ "coords", json::array() }
	};

	for (int i = 0; i < js_read["shapes"].size(); i++) {
		//Extraction et conversion des points
		std::vector<cv::Point2f> original_points, transformed_points;
		for (const auto& point : js_read["shapes"][i]["points"]) {
			original_points.emplace_back(cv::Point2f(point[0], point[1]));
		}

		//Application du warp sur les annotations, sauf pour le tableau (on le détecte avant le warp qui sert uniquement à lire son contenu)
		if (js_read["shapes"][i]["label"] != "board") {
			cv::perspectiveTransform(original_points, transformed_points, transform_matrix);
			//Application du resize sur les annotations
			for (auto& point : transformed_points) {
				point.x *= scale;
				point.y *= scale;
			}
		}
		else {
			transformed_points = original_points;
		}
		


		//Insertion des nouvelles coordonnées dans le json
		js["labels"].insert(js["labels"].end(), js_read["shapes"][i]["label"]);
		json json_arr = json::array({
			static_cast<int>(transformed_points[0].x + 0.5),
			static_cast<int>(transformed_points[0].y + 0.5),
			static_cast<int>(transformed_points[1].x + 0.5),
			static_cast<int>(transformed_points[1].y + 0.5),
			static_cast<int>(transformed_points[2].x + 0.5),
			static_cast<int>(transformed_points[2].y + 0.5),
			static_cast<int>(transformed_points[3].x + 0.5),
			static_cast<int>(transformed_points[3].y + 0.5) });
		js["coords"].insert(js["coords"].end(), json_arr);

		/**
		//Dessin des polygones
		std::vector<cv::Point> points;
		for (const auto& point : transformed_points) {
			points.emplace_back(cv::Point(static_cast<int>(point.x + 0.5), static_cast<int>(point.y + 0.5)));
		}
		const cv::Point* pts = points.data();
		int npts = points.size();
		cv::polylines(warped_board, &pts, &npts, 1, true, cv::Scalar(0, 0, 255), 1);
		**/
	}

	cv::imwrite("C:/Users/scott/OneDrive/Bureau/images_slides/99.jpg", warped_board);
	return js;
}


//Calcul de l'IOU (instersection / union) pour deux objets
//a liste contenant les coordonnées de l'objet a
//b liste contenant les coordonnées de l'objet b
double intersection_over_union(json a, json b) {

	//Extraction et conversion des coordonnées des json
	std::vector<cv::Point2f> quad1(4), quad2(4);
	for (int i = 0; i < 4; i++) {
		quad1[i] = cv::Point2f(static_cast<float>(a[i * 2].get<double>()), static_cast<float>(a[i * 2 + 1].get<double>()));
		quad2[i] = cv::Point2f(static_cast<float>(b[i * 2].get<double>()), static_cast<float>(b[i * 2 + 1].get<double>()));
	}

	//Calcul de l'intersection des coordonnées des objets détéctés avec les annotations
	std::vector<cv::Point2f> intersection_polygon;
	cv::intersectConvexConvex(quad1, quad2, intersection_polygon);

	//renvoyer 0.0 quand il n'y a pas d'intersection
	if (intersection_polygon.empty()) {
		return 0.0;
	}

	//Contours de l'intersection
	double intersection_area = cv::contourArea(intersection_polygon);

	//Contours de l'union
	double union_area = cv::contourArea(quad1) + cv::contourArea(quad2) - intersection_area;

	//Calcul de l'IOU
	return intersection_area / union_area;
}





//Fonction qui retourne le nombre de vrai positifs, faux positifs, faux négatifs et l'iou moyenne pour chaque classe
//d = json qui contient les objets détéctés par le programme
//g = json qui contient les vérités terrain (donné par le json créé avec labelme)
std::tuple<int, double, int, double, int, double, int, double, int, int, int, int, int, int, int, int> tp_fp_fn(json g, json d, double iou_threshold) {
	double tp_board = 0, tp_board_mean_iou = 0, tp_line = 0, tp_line_mean_iou = 0, tp_word = 0, tp_word_mean_iou = 0, tp_letter = 0, tp_letter_mean_iou = 0;
	int fp_board = 0, fp_line = 0, fp_word = 0, fp_letter = 0;
	int lab_board = 0, lab_line = 0, lab_word = 0, lab_letter = 0;
	int fn_board = 0, fn_line = 0, fn_word = 0, fn_letter = 0;


	//parcours du json des objets détéctés
	for (int i = 0; i < d["labels"].size(); i++) {
		double best_iou = 0;
		int idx_best_iou = 0;

		//parcours du json des vérités terrain
		for (int j = 0; j < g["labels"].size(); j++) {

			//Calcul de l'intersection sur l'union
			double iou = intersection_over_union(d["coords"][i], g["coords"][j]);

			//Si les labels correspondents et si l'iou est plus grandre que la précédente, on la garde
			if (d["labels"][i] == g["labels"][j] && iou > best_iou) {
				best_iou = iou;
				idx_best_iou = j;
			}
		}

		//Si l'iou est supérieure au seuil, on incrémente le nombre de vrais positifs du label associé
		if (best_iou >= iou_threshold) {
			if (g["labels"][idx_best_iou] == "board") {
				tp_board += 1;
				tp_board_mean_iou += best_iou;
			}
			if (g["labels"][idx_best_iou] == "line") {
				tp_line += 1;
				tp_line_mean_iou += best_iou;
			}
			if (g["labels"][idx_best_iou] == "word") {
				tp_word += 1;
				tp_word_mean_iou += best_iou;
			}
			if (g["labels"][idx_best_iou] == "letter") {
				tp_letter += 1;
				tp_letter_mean_iou += best_iou;
			}
			g["coords"].erase(g["coords"].begin() + idx_best_iou);
			g["labels"].erase(g["labels"].begin() + idx_best_iou);
		}
		//Sinon on incrémente le nombre de faux positifs du label associé
		else {
			if (d["labels"][i] == "board") {
				fp_board += 1;
			}
			if (d["labels"][i] == "line") {
				fp_line += 1;
			}
			if (d["labels"][i] == "word") {
				fp_word += 1;
			}
			if (d["labels"][i] == "letter") {
				fp_letter += 1;
			}
		}
	}
	
	//Pour chaque anotation qui n'a pas été associée à un objet détécté, on incrémente le nombre de faux négatifs pour le label associé
	for (int i = 0; i < g["labels"].size(); i++) {
		if (g["labels"][i] == "board") {
			fn_board += 1;
		}
		else if (g["labels"][i] == "line") {
			fn_line += 1;
		}
		else if (g["labels"][i] == "word") {
			fn_word += 1;
		}
		else if (g["labels"][i] == "letter") {
			fn_letter += 1;
		}
	}
	
	if (tp_board != 0) {
		tp_board_mean_iou /= tp_board;
	}
	if (tp_line != 0) {
		tp_line_mean_iou /= tp_line;
	}
	if (tp_word != 0) {
		tp_word_mean_iou /= tp_word;
	}
	if (tp_letter != 0) {
		tp_letter_mean_iou /= tp_letter;
	}

	return std::make_tuple(tp_board, tp_board_mean_iou, tp_line, tp_line_mean_iou, tp_word, tp_word_mean_iou, tp_letter, tp_letter_mean_iou,
		fp_board, fp_line, fp_word, fp_letter,
		fn_board, fn_line, fn_word, fn_letter);
}



//evalutation pour une image
void evaluation(json g, json d, double iou_threshold) {

	auto tp = tp_fp_fn(g, d, iou_threshold);
	int tp_board = std::get<0>(tp), tp_line = std::get<2>(tp), tp_word = std::get<4>(tp), tp_letter = std::get<6>(tp);
	double tp_board_mean_iou = std::get<1>(tp), tp_line_mean_iou = std::get<3>(tp), tp_word_mean_iou = std::get<5>(tp), tp_letter_mean_iou = std::get<7>(tp);
	int fp_board = std::get<8>(tp), fp_line = std::get<9>(tp), fp_word = std::get<10>(tp), fp_letter = std::get<11>(tp);
	int fn_board = std::get<12>(tp), fn_line = std::get<13>(tp), fn_word = std::get<14>(tp), fn_letter = std::get<15>(tp);


	double precision_board = (double)tp_board / (tp_board + fp_board);
	double recall_board = (double)tp_board / (tp_board + fn_board);
	double f1_score_board = (2 * precision_board * recall_board) / (precision_board + recall_board);

	double precision_line = (double)tp_line / (tp_line + fp_line);
	double recall_line = (double)tp_line / (tp_line + fn_line);
	double f1_score_line = (2 * precision_line * recall_line) / (precision_line + recall_line);

	double precision_word = (double)tp_word / (tp_word + fp_word);
	double recall_word = (double)tp_word / (tp_word + fn_word);
	double f1_score_word = (2 * precision_word * recall_board) / (precision_word + recall_board);

	double precision_letter = (double)tp_letter / (tp_letter + fp_letter);
	double recall_letter = (double)tp_letter / (tp_letter + fn_letter);
	double f1_score_letter = (2 * precision_letter * recall_letter) / (precision_letter + recall_letter);

	double average_precision = (precision_board + precision_line + precision_word + precision_letter) / 4;
	double average_recall = (recall_board + recall_line + recall_board + recall_letter) / 4;
	double average_f1_score = (2 * average_precision * average_recall) / (average_precision + average_recall);
	double tp_mean_iou = (tp_board_mean_iou, tp_line_mean_iou, tp_word_mean_iou, tp_letter_mean_iou) / 4;


	std::cout << "board" << "	Precision: " << precision_board << "	Recall: " << recall_board << "	F1_score: " << f1_score_board << "	Average iou of true positives: " << tp_board_mean_iou << std::endl << std::endl;

	std::cout << "line" << "	Precision: " << precision_line << "	Recall: " << recall_line << "	F1_score: " << f1_score_line << "	Average iou of true positives: " << tp_line_mean_iou << std::endl << std::endl;

	std::cout << "word" << "	Precision: " << precision_word << "	Recall: " << recall_word << "	F1_score: " << f1_score_word << "	Average iou of true positives: " << tp_word_mean_iou << std::endl << std::endl;

	std::cout << "letter" << "	Precision: " << precision_letter << "	Recall: " << recall_letter << "	F1_score: " << f1_score_letter << "	Average iou of true positives: " << tp_letter_mean_iou << std::endl << std::endl;

	std::cout << "mean" << "	Average precision: " << average_precision << "	Average recall: " << average_recall << "	Average f1-score: " << average_f1_score << std::endl << std::endl;

	
	std::cout << "tp_board: " << tp_board << std::endl;
	std::cout << "tp_line: " << tp_line << std::endl;
	std::cout << "tp_word: " << tp_word << std::endl;
	std::cout << "tp_letter: " << tp_letter << std::endl;
	
	std::cout << "fp_board: " << fp_board << std::endl;
	std::cout << "fp_line: " << fp_line << std::endl;
	std::cout << "fp_word: " << fp_word << std::endl;
	std::cout << "fp_letter: " << fp_letter << std::endl;

	std::cout << "fn_board: " << fn_board << std::endl;
	std::cout << "fn_line: " << fn_line << std::endl;
	std::cout << "fn_word: " << fn_word << std::endl;
	std::cout << "fn_letter: " << fn_letter << std::endl;
}



int main(){
	std::string folder_path = "C:/Users/scott/OneDrive/Bureau/images_poly/";//images_poly/;
	std::vector<cv::String> jpg_files, JPG_files;

	//On récupère les jpg et JPG
	cv::glob(folder_path + "*.jpg", jpg_files, true);
	cv::glob(folder_path + "*.JPG", JPG_files, true);

	//On combine les deux
	jpg_files.insert(jpg_files.end(), JPG_files.begin(), JPG_files.end());

	//Lancement du traitement sur toutes les images et json
	for (const auto& file : jpg_files) {

		//Lecture du json
		std::string json_path = file.substr(0, file.find_last_of('.')) + ".json";
		cv::String json_file(json_path);

		//Extraction du nom de l'image courante
		size_t last_separator = file.find_last_of("/\\");
		size_t last_dot = file.find_last_of('.');
		std::string image_name = file.substr(last_separator + 1, last_dot - last_separator - 1);


		std::cout << "Traitement de l'image : " << image_name << std::endl;

		//Detection
		auto detection_results = detection(file, image_name);
		json detected = std::get<0>(detection_results);
		cv::Mat transform_matrix = std::get<1>(detection_results);
		cv::Mat warped_board = std::get<2>(detection_results);
		int warped_board_width = std::get<3>(detection_results);
		int warped_board_height = std::get<4>(detection_results);
		float scale = std::get<5>(detection_results);

		//Simplification du json des annotations et application de la transformation sur les annotations
		json ground_truth = simplify_json_and_transformation(json_path, transform_matrix, warped_board_width, warped_board_height, warped_board, scale);

		//Evalutation
		evaluation(ground_truth, detected, 0.7);

		std::cout << std::endl << std::endl;
	}
	return 0;
}
