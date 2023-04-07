#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <C:/Users/scott/OneDrive/Bureau/opencv/json.hpp>//télécharger le fichier json nlohmann (pas besoin d'installation, juste include ce fichier)

using json = nlohmann::json;






//utilisation de labelme pour générer les json
//pour les tests, quand on entoure les objets, if faut faire un rectangle avec le premier point en haut à gauche et le deuxième en bas à droite (sinon l'évaluation ne marche pas)



/**
* Rotation de l'image en fonction de lignes de Hough détéctées dans l'image
* 
* @param img l'image à modifier
* @param closing l'image binaire servant à trouver les lignes de Hough (qu'on modifie aussi)
*/
void hough_rotate(cv::Mat& img, cv::Mat& closing) {
	cv::Mat dst, lines;
	cv::Canny(closing, dst, 50, 200, 3);
	cv::HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);

	double sum_theta = 0;
	int nb_values = 0;

	for (int i = 0; i < lines.rows; i++) {
		float t = lines.at<float>(cv::Point(1, i));
		if (t < 1) {
			sum_theta += t;
			nb_values += 1;
		}
	}
	double angle = 180 * (sum_theta / nb_values) / CV_PI;
	cv::Point2d center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::warpAffine(img, img, rotation_matrix, img.size());
	cv::warpAffine(closing, closing, rotation_matrix, closing.size());
}

/**
* Trie la liste de composantes connexes par rapport à l'aire (ordre décroissant) et séléction des n premiers éléments
* Pas de return, on modifie juste la matrice en paramètre
* 
* @param stats matrice contenant les composantes connexes, en particulier leurs coordonnées et aires
* @param b booléen qui indique si on veut séléctionner les n premiers éléments ou juste trier
* @param n entier qui indique le nombre de lignes de la matrice à récupérer
*/
void reverse_sort_by_fourth_column(cv::Mat& stats, bool b, int n) {
	cv::Mat col_4 = stats.col(4);
	cv::Mat1i idx;
	cv::sortIdx(col_4, idx, cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);
	cv::Mat result(stats.rows, stats.cols, CV_32S);
	for (int y = 0; y < stats.rows; y++) {
		stats.row(idx(y)).copyTo(result.row(y));
	}
	if (b) {
		if (result.rows > n) {
			stats = result.rowRange(1, n);
		}
	}
	else {
		stats = result;
	}
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
		if (col_index >= 0 && col_index < mat.cols) { // add bounds check
			for (int j = 0; j < mat.rows; j++) {
				mat.at<char>(j, col_index) = 0;
			}
		}
	}
}



/**
* segmentation du tableau, crop et segmentation du texte dans les tableaux
* @param path chemin de l'image du tableau
* @return un dictionnaire json qui contient deux clés "labels" et "coords" qui contiennent les listes des coordonnées et des labels
*/
json detection(std::string path) {

	//Conversion en hsv -> flou gaussien -> seuillage (sélection des pixels verts)
	cv::Mat hsv, gaussian, binary, closing, img = cv::imread(path);
	//cv::resize(img, img, cv::Size(1000, 1000), cv::INTER_LINEAR);
	cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
	cv::GaussianBlur(hsv, gaussian, cv::Size(3, 3), 0);
	cv::inRange(gaussian, cv::Scalar(30, 0, 0), cv::Scalar(120, 255, 170), binary);

	//Fermeture binaire
	cv::Mat kernel = cv::Mat::ones(cv::Size(5, 5), CV_8UC1);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::morphologyEx(binary, closing, cv::MORPH_CLOSE, kernel);

	//Recherche des contours, sélection du plus grand (le tableau)
	cv::findContours(closing, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	
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
	cv::approxPolyDP(contours[largest_area_index], approx, 0.1 * cv::arcLength(contours[largest_area_index], true), true);



	std::vector<std::vector<cv::Point>> contours_poly(1);
	contours_poly[0] = approx;

	//drawContours(img, contours_poly, 0, cv::Scalar(0, 0, 255), 2);

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

	cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
	cv::fillPoly(mask, contours_poly, cv::Scalar(255));
	cv::Mat outputImage = cv::Mat::zeros(img.size(), img.type());
	img.copyTo(outputImage, mask);


	//Crop du tableau
	cv::Mat cropped_board = outputImage(cv::Range(y_min, y_max), cv::Range(x_min, x_max));



	/*

	cv::Rect bounding_rect = cv::boundingRect(approx);
	cv::Mat cropped_image = img(bounding_rect);


	// Compute moments of the contour
	cv::Moments moments = cv::moments(contours[largest_area_index]);

	// Compute centroid of the contour
	cv::Point2f centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

	// Sort points in 'approx' in clockwise order around the centroid
	std::sort(approx.begin(), approx.end(), [centroid](cv::Point a, cv::Point b) {
		return std::atan2(a.y - centroid.y, a.x - centroid.x) < std::atan2(b.y - centroid.y, b.x - centroid.x);
		});


	std::vector<cv::Point2f> src_points;
	src_points.push_back(approx[0]);
	src_points.push_back(approx[1]);
	src_points.push_back(approx[2]);
	src_points.push_back(approx[3]);

	std::vector<cv::Point2f> dst_points;
	dst_points.push_back(cv::Point2f(0, 0));
	dst_points.push_back(cv::Point2f(img.cols, 0));
	dst_points.push_back(cv::Point2f(img.cols, img.rows));
	dst_points.push_back(cv::Point2f(0, img.rows));

	cv::Mat transform_matrix = cv::getPerspectiveTransform(src_points, dst_points);
	cv::Mat cropped_board;
	cv::warpPerspective(img, cropped_board, transform_matrix, cv::Size(img.cols, img.rows));





	cv::imshow("cropped_board", cropped_board);
	cv::waitKey(0);

	*/




	//hough_rotate(img, closing); #remplacer par warp


	//Création de l'objet json qui va contenir tous les labels détectés:
	//"labels" : ["board", "board", "line", "line", "line", "word", "letter", etc]
	//"coords": [[xmin, ymin, w, h], [xmin, ymin, w, h], [xmin, ymin, w, h], etc]
	json js = {
		{"labels", json::array()},
		{"coords", json::array()}
	};
	js["coords"].insert(js["coords"].end(), json::array({ x_min, y_min, x_max, y_max}));
	js["labels"].insert(js["labels"].end(), "board");

	//cv::rectangle(img, cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min), cv::Scalar(0, 0, 255), 2);


	//Conversion du tableau croppé en gris, puis seuillage, puis dilatation binaire pour coller les lettres qui appartiennent aux mêmes mots
	cv::Mat closing_kernel_word = cv::Mat::ones(cv::Size(50, 1), CV_8UC1);
	cv::Mat word_labels, word_stats, word_centroids, gray_board, binary_board, closed_board;

	cv::cvtColor(cropped_board, gray_board, cv::COLOR_BGR2GRAY);
	cv::threshold(gray_board, binary_board, 180, 255, cv::THRESH_BINARY);
	cv::morphologyEx(binary_board, closed_board, cv::MORPH_CLOSE, closing_kernel_word);//Tester fermeture binaire

	//Recherche des composantes connexes (lignes)
	cv::connectedComponentsWithStats(closed_board, word_labels, word_stats, word_centroids);


	//reverse_sort_by_fourth_column(text_stats, true, 1);//appliquer seuillage ratio

	std::vector<std::vector<int>> word_bboxes;

	//Ajout des labels et coordonnées correspondantes aux lignes trouvées
	for (int k = 1; k < word_stats.rows; k++) {

		int x = word_stats.at<int>(cv::Point(0, k)) + x_min;
		int y = word_stats.at<int>(cv::Point(1, k)) + y_min;
		int w = word_stats.at<int>(cv::Point(2, k));
		int h = word_stats.at<int>(cv::Point(3, k));
		if (w*h > 500){//voir en fonction des images
			
			cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
			js["coords"].insert(js["coords"].end(), json::array({ x, y, x + w, y + h }));
			js["labels"].insert(js["labels"].end(), "word");


			//Suppression des lettres des autres mots qui dépassent sur le mot courant
			cv::Mat binary_cropped_closed_word = closed_board(cv::Rect(x - x_min, y - y_min, w, h));
			cv::Mat labs, stats, cent;
			int numLabels = cv::connectedComponentsWithStats(binary_cropped_closed_word, labs, stats, cent);

			int largestLabel = 0;
			int largestSize = 0;
			for (int i = 1; i < numLabels; i++) {
				int size = stats.at<int>(i, cv::CC_STAT_AREA);
				if (size > largestSize) {
					largestLabel = i;
					largestSize = size;
				}
			}

			cv::Mat binary_cropped_word = binary_board(cv::Rect(x - x_min, y - y_min, w, h));
			cv::Mat largestComponentMask = (labs == largestLabel);
			cv::Mat result;
			cv::bitwise_and(binary_cropped_word, largestComponentMask, result);


			//Ajout des bbox des mots dans un vecteur pour pouvoir trouver les lignes (vers la fin de la fonction)
			word_bboxes.emplace_back(std::vector<int>{x, y, w, h});

			//Range: seuil des images sur lesquelles appliquer la détection de lettres
			int range = 60;

			if (result.cols > range) {

				int width = result.cols;
				int height = 1;

				//Calcul de l'histogramme projeté vertical
				cv::Mat hist = cv::Mat::zeros(height, width, CV_32FC1);
				cv::reduce(result, hist, 0, cv::REDUCE_SUM, CV_32FC1);

				//Normalisation de l'histogramme (surtout pour la visualisation)
				double min_val, max_val;
				cv::minMaxLoc(hist, &min_val, &max_val);

				hist = hist / max_val * 255;

				hist.convertTo(hist, CV_8UC1);


				//Appel de la fonction permettant de trouver les minimums locaux
				//Param 1: histogramme
				//Param 2: fenêtre sur laquelle chercher chaque minimum
				std::vector<int> minima = find_local_minima(hist, range);//voir avec flou gaussien


				/*
				for (int i = 0; i < minima.size(); i++) {
					std::cout << static_cast<int>(minima[i]) << " ";
				}*/


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
					if (wl*hl >= 200) {//On ne prend pas les éléments avec une aire trop petite
						cv::rectangle(img, cv::Rect(xl, yl, wl, hl), cv::Scalar(0, 0, 0));
						js["coords"].insert(js["coords"].end(), json::array({ xl, yl, xl + wl, yl + hl }));
						js["labels"].insert(js["labels"].end(), "letter");
					}
				}
			}
		}
	}

	/*
	for (const auto& subvector : word_bboxes) {
		std::cout << "xmin: " << subvector[0] << ", ymin: " << subvector[1] << ", w: " << subvector[2] << ", h: " << subvector[3] << std::endl;
	}*/


	//Les mots détéctés avec la fonction cv::connectedComponentsWithStats sont triés par défaut en fonction de leur valeur y_min
	//Pour chaque bounding box des mots, on la compare à celle d'avant. Si elle est à peu près à la même hauteur, il appartient à la même ligne
	//Sinon, on crée une nouvelle ligne et on recommence (on peut créer les lignes au fur et à mesure sans retourner en arrière comme les bbox sont déjà triées par y_min
	std::vector<std::vector<int>> lines;
	lines.emplace_back(std::vector<int>{0});

	for (int i = 1; i < word_bboxes.size(); i++) {
		if (((word_bboxes[i - 1][1] + word_bboxes[i - 1][3]) <= word_bboxes[i][1]) || (word_bboxes[i - 1][1] >= (word_bboxes[i][1] + word_bboxes[i][3]))){
			lines.emplace_back(std::vector<int>{i});
		}
		else if ((word_bboxes[i - 1][1] > word_bboxes[i][1]) && ((word_bboxes[i - 1][1] + word_bboxes[i - 1][3]) < (word_bboxes[i][1] + word_bboxes[i][3])) || (word_bboxes[i - 1][1] < word_bboxes[i][1]) && ((word_bboxes[i - 1][1] + word_bboxes[i - 1][3]) > (word_bboxes[i][1] + word_bboxes[i][3]))) {
			lines.back().push_back(i);
		}
		else if (word_bboxes[i - 1][1] < word_bboxes[i][1]) {
			int dist = word_bboxes[i][1] + word_bboxes[i][3] - word_bboxes[i - 1][1];
			double vertical_intersection = std::max((double)dist / word_bboxes[i][3], (double)dist / word_bboxes[i-1][3]);
			if (vertical_intersection >= 0.7) {
				lines.back().push_back(i);
			}
		}
		else if (word_bboxes[i - 1][1] > word_bboxes[i][1]) {
			int dist = word_bboxes[i - 1][1] + word_bboxes[i - 1][3] - word_bboxes[i][1];
			double vertical_intersection = std::max((double)dist / word_bboxes[i][3], (double)dist / word_bboxes[i-1][3]);
			if (vertical_intersection >= 0.7) {
				lines.back().push_back(i);
			}
		}
	}

	//tri des mots de chaque ligne en fonction de leur x_min, pour les remettre dans l'ordre
	for (auto& line : lines) {
		std::sort(line.begin(), line.end(), [&](int i, int j) {
			return word_bboxes[i][0] < word_bboxes[j][0];
			});
	}

	//print de l'arrangement des mots détéctés (juste pour avoir un aperçu, 0 = première composante connexe trouvée, par ordre y_min)
	for (const auto& subvector : lines) {
		for (const auto& val : subvector) {
			std::cout << val << "   ";
		}
		std::cout << std::endl << std::endl;
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


		cv::rectangle(img, cv::Rect(xmin_line, ymin_line, w_line, h_line), cv::Scalar(0, 255, 0));

		js["coords"].insert(js["coords"].end(), json::array({ xmin_line, ymin_line, xmin_line + w_line, ymin_line + h_line }));
		js["labels"].insert(js["labels"].end(), "line");
	}
	//TODO: enlever les composantes connexes trop petites pour les lettres


	//cv::imshow("Image", img);
	//cv::waitKey(0);
	cv::imwrite("C:/Users/scott/OneDrive/Bureau/output/output.jpg", img);
	return js;
}



/**
* Simplifie le json généré par labelme donné en paramètre (enlève toutes les informations inutiles)
* @param path chemin du json
* @return nouveau json siomplifié
*/
json simplify_json(std::string path) {
	std::ifstream f(path);
	json js_read = json::parse(f);
	json js = {
		{ "labels",json::array() },
		{ "coords", json::array() }
	};
	for (int i = 0; i < js_read["shapes"].size(); i++) {
		js["labels"].insert(js["labels"].end(), js_read["shapes"][i]["label"]);
		json json_arr = json::array({
			static_cast<int>(js_read["shapes"][i]["points"][0][0] + 0.5),
			static_cast<int>(js_read["shapes"][i]["points"][0][1] + 0.5),
			static_cast<int>(js_read["shapes"][i]["points"][1][0] + 0.5),
			static_cast<int>(js_read["shapes"][i]["points"][1][1] + 0.5) });
		js["coords"].insert(js["coords"].end(), json_arr);
	}
	return js;
}

/**
* Calcule l'intersection sur l'union pour deux objets
* @param a liste contenant les coordonnées de l'objet a
* @param b liste contenant les coordonnées de l'objet b
* @return la valeur de l'iou au format double
**/
double intersection_over_union(json a, json b) {
	cv::Rect rect1((int)a[0], (int)a[1], (int)a[2] - (int)a[0], (int)a[3] - (int)a[1]);
	cv::Rect rect2((int)b[0], (int)b[1], (int)b[2] - (int)b[0], (int)b[3] - (int)b[1]);
	cv::Rect rect_intersect = rect1 & rect2;
	cv::Rect rect_union = rect1 | rect2;
	double inter_over_union = (double)rect_intersect.area() / (double)rect_union.area();
	return inter_over_union;
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
			if (d["labels"][i] == g["labels"][j] && iou > best_iou) {
				best_iou = iou;
				idx_best_iou = j;
			}
		}
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



//evalutation pour une seule image
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


	std::cout << "board" << std::endl;
	std::cout << "	Precision: " << precision_board << "	Recall: " << recall_board << "	F1_score: " << f1_score_board << "	Average iou of true positives: " << tp_board_mean_iou << std::endl << std::endl;

	std::cout << "line" << std::endl;
	std::cout << "	Precision: " << precision_line << "	Recall: " << recall_line << "	F1_score: " << f1_score_line << "	Average iou of true positives: " << tp_line_mean_iou << std::endl << std::endl;

	std::cout << "word" << std::endl;
	std::cout << "	Precision: " << precision_word << "	Recall: " << recall_word << "	F1_score: " << f1_score_word << "	Average iou of true positives: " << tp_word_mean_iou << std::endl << std::endl;

	std::cout << "letter" << std::endl;
	std::cout << "	Precision: " << precision_letter << "	Recall: " << recall_letter << "	F1_score: " << f1_score_letter << "	Average iou of true positives: " << tp_letter_mean_iou << std::endl << std::endl;

	std::cout << "mean" << std::endl;
	std::cout << "	Average precision: " << average_precision << "	Average recall: " << average_recall << "	Average f1-score: " << average_f1_score << "	tp mean iou: " << tp_mean_iou << std::endl << std::endl;


	
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



int main() {
	/*
	std::vector<cv::String> files;
	cv::glob("C:/Users/scott/onedrive/bureau/*.json", files);
	//std::string str = path.substr(0, str.size() - 4);
	//std::string path = "C:/Users/scott/OneDrive/Bureau/projet image m1/tab1.jpeg";
	*/


	std::string path = "C:/Users/scott/OneDrive/Bureau/images/18.jpg";
	std::string path1 = "C:/Users/scott/OneDrive/Bureau/images/18.json";

	json ground_truth = simplify_json(path1);
	//std::cout << ground_truth << std::endl;

	json detected = detection(path);
	//std::cout << detected << std::endl;

	evaluation(ground_truth, detected, 0.7);
	return 0;
}
