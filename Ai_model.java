package edu.poly.student.model;


import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.Pattern;

import org.springframework.web.multipart.MultipartFile;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Ai_model {

		@NotEmpty(message="ID không dược để trống!")
		@Pattern(regexp = "\\d+", message = "ID phải là chữ số")
		private String id;
		//lưu trữ file ảnh được upload từ form.
		//MultipartFile là interface của Spring Framework được sử dụng để xử lý file upload trong các ứng dụng web. 
		//Interface này cung cấp các phương thức để lấy thông tin về file được upload như tên file, kích thước, kiểu MIME, đường dẫn tạm thời
		private MultipartFile imageFile;
		private String imageUrl;
	

}
