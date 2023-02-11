#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <C:/Users/scott/OneDrive/Bureau/opencv/json.hpp>//télécharger le fichier json nlohmann (pas besoin d'installation, juste include ce fichier)

using json = nlohmann::json;






//utilisation de labelme pour générer les json
//pour les tests, quand on entoure les objets, if faut faire un rectangle avec le premier point en haut à gauche et le deuxième en bas à droite (sinon le code marche pas)






//TODO: ignorer les zones vides après rotation
//TODO: évaluation json + position, quantité, type



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

/**
* segmentation du tableau, crop et segmentation du texte dans les tableaux
* @param path chemin de l'image du tableau
* @return un dictionnaire json qui contient deux clés "labels" et "coords" qui contiennent les listes des coordonnées et des labels
*/
json detection(std::string path) {

	cv::Mat gray, gaussian, binary, closing, img = cv::imread(path);
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gray, gaussian, cv::Size(11, 11), 0);
	cv::adaptiveThreshold(gaussian, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 5);

	cv::Mat kernel = cv::Mat::ones(cv::Size(5, 5), CV_8UC1);
	cv::morphologyEx(binary, closing, cv::MORPH_CLOSE, kernel);
	cv::bitwise_not(closing, closing);//Vérifier en fonction des photos

	hough_rotate(img, closing);

	cv::Mat labels, stats, centroids;
	cv::connectedComponentsWithStats(closing, labels, stats, centroids);
	reverse_sort_by_fourth_column(stats, true, 5);

	json js = {
		{"labels", json::array()},
		{"coords", json::array()}
	};

	std::vector<std::vector<int>> coords(stats.rows, std::vector<int>(stats.cols));
	std::vector<cv::Mat> cropped_imgs;
	for (int i = 0; i < stats.rows; i++) {
		int x = stats.at<int>(cv::Point(0, i));
		int y = stats.at<int>(cv::Point(1, i));
		int w = stats.at<int>(cv::Point(2, i));
		int h = stats.at<int>(cv::Point(3, i));
		cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255));
		cropped_imgs.insert(cropped_imgs.end(), gaussian(cv::Range(y, y + h), cv::Range(x, x + w)));
		js["coords"].insert(js["coords"].end(), json::array({ x, y, x + w, y + h }));
		js["labels"].insert(js["labels"].end(), "board");

		coords[i][0] = x;
		coords[i][1] = y;
	}

	cv::Mat kernel_text = cv::Mat::ones(cv::Size(31, 1), CV_8UC1);
	for (int i = 0; i < cropped_imgs.size(); i++) {
		cv::Mat text_binary, text_closing, text_labels, text_stats, text_centroids;
		cv::adaptiveThreshold(cropped_imgs[i], text_binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 5);
		cv::bitwise_not(text_binary, text_binary);//vérifier en fonctions des photos
		cv::morphologyEx(text_binary, text_closing, cv::MORPH_CLOSE, kernel_text);
		cv::connectedComponentsWithStats(text_closing, text_labels, text_stats, text_centroids);

		reverse_sort_by_fourth_column(text_stats, false, 0);

		for (int k = 0; k < text_stats.rows; k++) {
			int x = text_stats.at<int>(cv::Point(0, k)) + coords[i][0];
			int y = text_stats.at<int>(cv::Point(1, k)) + coords[i][1];
			int w = text_stats.at<int>(cv::Point(2, k));
			int h = text_stats.at<int>(cv::Point(3, k));
			if ((double)w / h <= 0.3 || (double)w / h >= 6) {
				cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0));
				js["coords"].insert(js["coords"].end(), json::array({ x, y, x + w, y + h }));
				js["labels"].insert(js["labels"].end(), "text");
			}
		}
	}
	cv::imshow("Image", img);
	cv::waitKey(0);
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
double intersection_over_union(json a, json b) {
	cv::Rect rect1((int)a[0], (int)a[1], (int)a[2] - (int)a[0], (int)a[3] - (int)a[1]);
	cv::Rect rect2((int)b[0], (int)b[1], (int)b[2] - (int)b[0], (int)b[3] - (int)b[1]);
	cv::Rect rect_intersect = rect1 & rect2;
	cv::Rect rect_union = rect1 | rect2;
	double inter_over_union = (double)rect_intersect.area() / (double)rect_union.area();
	return inter_over_union;
}

/********************************************************************************************************************************************************************************/
/*****************************************************************pas fini*******************************************************************************************************/

//TODO: renvoyer le dictionnaire contenant le reste
void true_positives(json g, json d, double iou_threshold, int& tp_board, int& tp_text, double& tp_board_mean_iou, double& tp_text_mean_iou) {
	json rest;
	for (int i = 0; i < g["labels"].size(); i++) {
		double best_iou = 0;
		int idx = 0;
		for (int j = 0; j < d["labels"].size(); j++) {
			double iou = intersection_over_union(g["coords"][i], d["coords"][j]);
			if (g["labels"][i] == d["labels"][j] && iou > iou_threshold && iou > best_iou) {
				best_iou = iou;
				idx = j;
			}
		}
		if (best_iou != 0) {
			if (g["labels"][i] == "board") {
				tp_board += 1;
				tp_board_mean_iou += best_iou;

			}
			else {
				tp_text += 1;
				tp_text_mean_iou += best_iou;
			}
		}
	}
	if (tp_board != 0) {
		tp_board_mean_iou /= tp_board;
	}
	if (tp_board != 0) {
		tp_text_mean_iou /= tp_text;
	}
}


void false_positives(json g, json d, double iou_threshold, int& fp_board_wrong, int& fp_text_wrong, int& fp_background_wrong_board, int& fp_background_wrong_text) {
	for (int i = 0; i < d["labels"].size(); i++) {
		bool object_found = false;
		bool best_iou_wrong_class = false;
		for (int j = 0; j < g["labels"].size(); j++) {
			double iou = intersection_over_union(g["coords"][j], d["coords"][i]);
			if (iou > iou_threshold) {
				object_found = true;
				if (d["labels"][i] != g["labels"][j]) {
					if (d["labels"][i] == "board") {
						fp_board_wrong += 1;
					}
					else {
						fp_text_wrong += 1;
					}
				}
			}
		}
		if (object_found == false) {
			if (d["labels"][i] == "board") {
				fp_background_wrong_board += 1;
			}
			else {
				fp_background_wrong_text += 1;
			}
		}
	}
}

void false_negatives(int& fp_board_wrong, int& fp_text_wrong, int& fp_background_wrong_board, int& fp_background_wrong_text) {
	//faux négatifs pour une classe = faux positifs de l'autre classe
}

//evalutation pour une seule image
void evaluation(json g, json d, double iou_threshold) {
	//precision = vp/(vp+fp)
	//recall = vp/(vp+fn)
	int tp_board = 0, tp_text = 0, fp_board_wrong = 0, fp_text_wrong = 0, fp_background_wrong_board = 0, fp_background_wrong_text = 0;
	double tp_board_mean_iou = 0, tp_text_mean_iou = 0;
	
	true_positives(g, d, iou_threshold, tp_board, tp_text, tp_board_mean_iou, tp_text_mean_iou);
	false_positives(g, d, iou_threshold, fp_board_wrong, fp_text_wrong, fp_background_wrong_board, fp_background_wrong_text);


	std::cout << "tp_board: " << tp_board << std::endl;
	std::cout << "tp_text: " << tp_text << std::endl;
	std::cout << "fp_wrong_class_board: " << fp_board_wrong << std::endl;
	std::cout << "fp_wrong_class_text: " << fp_text_wrong << std::endl;
	std::cout << "fp_wrong_board: " << fp_background_wrong_board << std::endl;
	std::cout << "fp_wrong_text: " << fp_background_wrong_text << std::endl;
	std::cout << "tp_board_mean_iou: " << tp_board_mean_iou << std::endl;
	std::cout << "tp_text_mean_iou: " << tp_text_mean_iou << std::endl;

	//TODO: f1-score



}

void non_max_suppression() {

}


/********************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************/



int main() {
	std::vector<cv::String> files;
	cv::glob("C:/Users/scott/onedrive/bureau/*.json", files);

	//std::string str = path.substr(0, str.size() - 4);

	std::string path = "C:/Users/scott/OneDrive/Bureau/projet image m1/tab1.jpeg";
	std::string path1 = "C:/Users/scott/OneDrive/Bureau/projet image m1/tab1.json";

	json ground_truth = simplify_json(path1);
	json detected = detection(path);


	evaluation(ground_truth, detected, 0.5);
	return 0;
}